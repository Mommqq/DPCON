import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm.auto import tqdm
import lpips
import time
from .utils import cosine_beta_schedule, extract, noise_like, default, linear_beta_schedule
import math
from torchvision.transforms import GaussianBlur
from pytorch_msssim import ms_ssim

class AutoBalanceWeight(nn.Module):
    def __init__(self, 
                 window_size: int = 7,
                 sigma: float = 1.5,
                 eps: float = 1e-6):
        super().__init__()
        # 高斯模糊参数
        self.gaussian_blur = GaussianBlur(5, sigma)
        # self.loss_fn_vgg = lpips.LPIPS(net="vgg", eval_mode=True)        
        # MS-SSIM 参数
        self.window_size = window_size
        self.eps = eps
        
        # 自适应权重参数
        self.register_buffer('ema_weight', torch.tensor(0.5))  # 指数移动平均

    def forward(self, orig, recon):
        """
        输入: 
            orig: 原始图像 (B,C,H,W) 
            recon: 生成图像 (B,C,H,W)
        输出:
            weight: 自适应权重 (标量)
        """
        # 预处理：高斯平滑消除高频噪声干扰
        orig = self.gaussian_blur(orig)
        recon = self.gaussian_blur(recon)
        
        # 计算多尺度结构相似性 (MS-SSIM)
        ms_ssim_value = ms_ssim(orig, recon, 
                                data_range=1.0,  # 假设输入图像值范围为 [0, 1]
                                win_size=self.window_size,
                                size_average=True)
        
        # 计算结构差异
        struct_diff = 1 - ms_ssim_value  # MS-SSIM 越小，结构差异越大
        # struct_diff = self.loss_fn_vgg(orig, recon).mean()
        # 计算像素级MSE
        mse = torch.mean((orig - recon)**2)
        
        # 平衡因子计算（避免除零）
        balance_factor = struct_diff / (mse + self.eps)
        # balance_factor = (mse+1-ms_ssim_value) / (struct_diff + self.eps)        
        # 动态权重生成（通过sigmoid约束到0-1范围）
        weight = torch.sigmoid(balance_factor - 1.0)  # 平衡因子阈值设为1
        
        # 使用EMA平滑权重变化
        self.ema_weight = 0.9 * self.ema_weight + 0.1 * weight.detach()
        
        return self.ema_weight

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入尺寸: (b, 3, 128, 256)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # -> (b, 64, 64, 128)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # -> (b, 128, 32, 64)
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # -> (b, 256, 16, 32)
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),           # -> (b, 512, 8, 16)
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 最后通过自适应池化将特征图压缩成固定大小，然后全连接映射到一个标量
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.conv_layers(x)  # (b, 512, 8, 16)
        # 自适应平均池化至 (1,1)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(x.size(0), -1)  # (b, 512)
        validity = self.fc(out)  # (b, 1)
        return validity.squeeze(1)  # 输出形状 (b,)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        context_fn,
        channels=3,
        num_timesteps=1000,
        loss_type="l1",
        clip_noise="half",
        vbr=False,
        lagrangian=1e-3,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=0,
        aux_loss_type="l1",
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.clip_noise = clip_noise
        self.vbr = vbr
        self.otherlogs = {}
        self.loss_type = loss_type
        self.lagrangian_beta = lagrangian
        self.var_schedule = var_schedule
        self.sample_steps = None
        self.aux_loss_weight = aux_loss_weight
        # self.aux_loss_weight = nn.Parameter(torch.tensor(0.1))

        self.aux_loss_type = aux_loss_type
        assert pred_mode in ["noise", "image", "renoise"]
        self.pred_mode = pred_mode
        to_torch = partial(torch.tensor, dtype=torch.float32)
        if aux_loss_weight > 0:
            self.loss_fn_vgg = lpips.LPIPS(net="vgg", eval_mode=False)
        else:
            self.loss_fn_vgg = None

        if var_schedule == "cosine":
            train_betas = cosine_beta_schedule(num_timesteps)
        elif var_schedule == "linear":
            train_betas = linear_beta_schedule(num_timesteps)
        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)
        # train_alphas_cumprod_prev = np.append(1.0, train_alphas_cumprod[:-1])
        (num_timesteps,) = train_betas.shape
        self.num_timesteps = int(num_timesteps)

        self.register_buffer("train_betas", to_torch(train_betas))
        self.register_buffer("train_alphas_cumprod", to_torch(train_alphas_cumprod))
        # self.register_buffer("train_alphas_cumprod_prev", to_torch(train_alphas_cumprod_prev))
        self.register_buffer("train_sqrt_alphas_cumprod", to_torch(np.sqrt(train_alphas_cumprod)))
        self.register_buffer(
            "train_sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - train_alphas_cumprod))
        )
        self.register_buffer(
            "train_sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / train_alphas_cumprod))
        )
        self.register_buffer(
            "train_sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / train_alphas_cumprod - 1))
        )


        # self.alpha = nn.Parameter(torch.tensor(math.pi/4))
        # self.alpha.data = 0.5*(1-torch.cos(2*self.alpha))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.alpha_mask = nn.Parameter(torch.randn(1, 3, 128, 256).to(device)) 
        self.alpha_mask.data = torch.sigmoid(self.alpha_mask)
        self.alpha = 1
        # self.D = Discriminator()
        self.W = AutoBalanceWeight()
        # self.alpha.data = torch.sigmoid(self.alpha)

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "loss_fn_vgg" not in name:
                yield param

    def get_extra_loss(self):
        return self.context_fn.get_extra_loss()

    def set_sample_schedule(self, sample_steps, device):
        self.sample_steps = sample_steps
        indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        self.alphas_cumprod = self.train_alphas_cumprod[indice]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = torch.sqrt(1.0 / self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sigma = torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        ) * torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_noise_train(self, x_t, t, noise):
        return (
            extract(self.train_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.train_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, x, t, context, clip_denoised):
        # noise = self.denoise_fn(x, self.sqrt_alphas_cumprod[t], context=context)
        if self.pred_mode == "noise":
            noise = self.denoise_fn(x, t.float().unsqueeze(-1) / self.sample_steps, context=context)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        else:
            x_recon = self.denoise_fn(
                x, t.float().unsqueeze(-1) / self.sample_steps, context=context
            )

        if clip_denoised == "full":
            x_recon.clamp_(-1.0, 1.0)
        elif clip_denoised == "half":
            x_recon[: x_recon.shape[0] // 2].clamp_(-1.0, 1.0)

        model_mean = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean

    def ddim(self, x, t, context,  clip_denoised, eta=0):
        noise = self.denoise_fn(x, t.float().unsqueeze(-1) / self.sample_steps, context=context)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        if clip_denoised == "full":
            x_recon.clamp_(-1.0, 1.0)
        elif clip_denoised == "half":
            x_recon[: x_recon.shape[0] // 2].clamp_(-1.0, 1.0)
        x_next = (
            extract(self.sqrt_alphas_cumprod_prev, t, x.shape) * x_recon
            + torch.sqrt(
                extract(self.one_minus_alphas_cumprod_prev, t, x.shape)
                - (eta * extract(self.sigma, t, x.shape)) ** 2
            )
            * noise + eta * extract(self.sigma, t, x.shape) * torch.randn_like(noise)
        )
        return x_next

    @torch.no_grad()
    def p_sample(self, x, t, context,clip_denoised, sample_mode="ddpm", eta=0):
        if sample_mode == "ddpm":
            model_mean = self.p_mean_variance(
                x=x, t=t, context=context, clip_denoised=clip_denoised
            )
            return model_mean
        elif sample_mode == "ddim":
            return self.ddim(x=x, t=t, context=context, clip_denoised=clip_denoised, eta=eta)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def p_sample_loop(self, shape, context,context_0,sample_mode, init=None, eta=0):
        device = self.alphas_cumprod.device

        b = shape[0]
        img = torch.zeros(shape, device=device) if init is None else init
        # buffer = []
        for count, i in enumerate(
            tqdm(
                reversed(range(0, self.sample_steps)),
                desc="sampling loop time step",
                total=self.sample_steps,
            )
        ):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                time,
                context=context,
                clip_denoised=self.clip_noise,
                sample_mode=sample_mode,
                eta=eta,
            )
        #     if count % 50 == 0:
        #         buffer.append(img)
        # buffer.append(img)
        # return img*self.alpha_mask+context_0[0]*(1-self.alpha_mask)
        return img*self.alpha+context_0[0]*(1-self.alpha)

    @torch.no_grad()
    def compress(
        self,
        images_x, 
        images_y,
        sample_steps=None,
        bitrate_scale=None,
        sample_mode="ddpm",
        bpp_return_mean=True,
        init=None,
        eta=0,
    ):
        context_dict = self.context_fn(images_x, images_y, bitrate_scale)
        # context_dict = self.context_fn(images_x, images_y)
        self.set_sample_schedule(
            self.num_timesteps if (sample_steps is None) else sample_steps,
            context_dict["output"][0].device,
            # context_dict["output_cond"][0].device,
        )
        return (
            self.p_sample_loop(
                images_x.shape, context_dict["output_cond"],context_dict["output"], sample_mode, init=init, eta=eta
            ),
#-------------------------------------------------------------------------------------------
            # self.p_sample_loop(
            #     images_x.shape, context_dict["output"],context_dict["output"], sample_mode, init=init, eta=eta
            # ),
            context_dict["bpp"].mean() if bpp_return_mean else context_dict["bpp"],
            context_dict["bpp_y"].mean() if bpp_return_mean else context_dict["bpp_y"],
            # context_dict["bpp_w"].mean() if bpp_return_mean else context_dict["bpp_w"]
        )

    def q_sample(self, x_start, t, noise):

        return (
            extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # def compute_gradient_penalty(D, real_samples, fake_samples):
    #     # 随机插值
    #     alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    #     interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    #     d_interpolates = D(interpolates)
        
    #     # 计算梯度
    #     gradients = torch.autograd.grad(
    #         outputs=d_interpolates,
    #         inputs=interpolates,
    #         grad_outputs=torch.ones_like(d_interpolates),
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True
    #     )[0]
        
    #     # 梯度惩罚项
    #     gradients = gradients.view(gradients.size(0), -1)
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #     return gradient_penalty

    # def huber_loss(self, x, delta=1.0):
    #     return torch.where(
    #         torch.abs(x) < delta,
    #         0.5 * x ** 2,
    #         delta * (torch.abs(x) - 0.5 * delta)
    #     )

    def p_losses(self, x_start, y_start, context_dict, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#---------------------------------------------------------------------------------------------
        fx = self.denoise_fn(
            x_noisy, t.float().unsqueeze(-1) / self.num_timesteps, context=context_dict["output_cond"]
        )
        # fx = self.denoise_fn(
        #     x_noisy, t.float().unsqueeze(-1) / self.num_timesteps, context=context_dict["output"]
        # )
        # x_recon = self.denoise_fn(
        #     x_noisy, self.train_sqrt_alphas_cumprod[t], context=context_dict["output"]
        # )

        if self.pred_mode == "noise":
            if self.loss_type == "l1":
                err = (noise - fx).abs().mean()
            elif self.loss_type == "l2":
                err = F.mse_loss(noise, fx)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError

        aux_err = 0

        if self.aux_loss_weight > 0:
            pred_x0 = self.predict_start_from_noise_train(x_noisy, t, fx).clamp(-1.0, 1.0)
            if self.aux_loss_type == "l1":
                aux_err = F.l1_loss(x_start, pred_x0)
            elif self.aux_loss_type == "l2":
                aux_err = F.mse_loss(x_start, pred_x0)
            elif self.aux_loss_type == "lpips":
                aux_err = self.loss_fn_vgg(x_start,pred_x0).mean()
                # aux_err = self.loss_fn_vgg(x_start, self.alpha*pred_x0+(1-self.alpha)*context_dict["output"][0]).mean()
                # aux_err = self.loss_fn_vgg(x_start, self.alpha_mask*pred_x0+(1-self.alpha_mask)*context_dict["output"][0]).mean()
            else:
                raise NotImplementedError()

            loss = ( 
                # self.lagrangian_beta * (context_dict["bpp"].mean()+context_dict["bpp_y"].mean()+0.003*context_dict["bpp_w"].mean())+F.mse_loss(y_start,context_dict["y_image"])
                self.lagrangian_beta * (context_dict["bpp"].mean())
                +F.mse_loss(y_start,context_dict["output_y"][0])
                # + 1-ms_ssim(y_start, context_dict["output"][0], data_range=1.0, size_average=True, win_size=7)
                # +F.mse_loss(y_start,context_dict["output_y_cond"][0])
                # +0.9*F.mse_loss(context_dict["output_cond"][0],context_dict["output"][0])
                # +F.mse_loss(context_dict["output"][0],x_start)
                # + err * (1 - self.aux_loss_weight)
                # + aux_err * self.aux_loss_weight

            )
            
            # loss = (self.d1*self.loss_fn_vgg(x_start,self.c1*pred_x0+(1-self.c1)*context_dict["output"][0]).mean()
            #         +(1-self.d1)*F.l1_loss(x_start,self.c1*pred_x0+(1-self.c1)*context_dict["output"][0])
            # )
            
            # maxLD = F.mse_loss(pred_x0,x_start)
            # minLD = F.mse_loss(context_dict["output"][0],x_start)
            # maxLP = self.loss_fn_vgg(x_start, context_dict["output"][0]).mean()
            # minLP = self.loss_fn_vgg(x_start, pred_x0).mean()
            # LD = F.mse_loss(self.alpha*pred_x0+(1-self.alpha)*context_dict["output"][0],x_start)
            
            LD = F.mse_loss(context_dict["output"][0],x_start)
            # LD = 1-ms_ssim(x_start, context_dict["output"][0], data_range=1.0, size_average=True, win_size=7)


            # LD = self.loss_fn_vgg(context_dict["output"][0],x_start).mean()
            
            # LP = aux_err+F.mse_loss(y_start,context_dict["output_y_cond"][0])
            # LP = aux_err*0.9+F.l1_loss(x_start, pred_x0)*0.1
            # d_real = self.D(x_start)
            # d_fake = self.D(pred_x0)
            # weight0 = (torch.mean(d_fake)/torch.mean(d_real))*(torch.mean(d_fake)/torch.mean(d_real))
            
            # d_loss = torch.mean(d_fake) - torch.mean(d_real)# 判别器损失
            
            # g_loss = -torch.mean(d_fake)                       # 生成器损失
            # mm = max(torch.mean(d_fake)/torch.mean(d_real),-torch.mean(d_fake)/torch.mean(d_real))
            # advLoss = 10*(1-(mm)**2)**2
            # weight = advLoss.detach()
            weight = self.W(x_start, pred_x0)
            # LP = 0.9*aux_err+F.l1_loss(x_start, pred_x0)*weight
            # LP = aux_err+F.l1_loss(x_start, pred_x0)*weight
            LP = aux_err
      
            dexLoss = 0.9*F.mse_loss(context_dict["output_cond"][0],x_start)+0.1*self.loss_fn_vgg(x_start,context_dict["output_cond"][0]).mean()
            # LP = (aux_err-minLP)/(maxLP-minLP)
            # LD = (F.mse_loss(self.alpha_mask*pred_x0+(1-self.alpha_mask)*context_dict["output"][0],x_start)-minLD)/(maxLD-minLD)
        else:
            loss = self.lagrangian_beta * context_dict["bpp"].mean() + err
        # loss = err.mean()

        return loss,LD,LP, weight, dexLoss

    def forward(self, images_x, images_y):
        device = images_x.device
        B, C, H, W = images_x.shape
    
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        if self.vbr:
            bitrate_scale = torch.rand(size=(B,), device=device)
            self.lagrangian_beta = self.scale_to_beta(bitrate_scale)
        else:
            bitrate_scale = None
        output_dict = self.context_fn(images_x, images_y, bitrate_scale)
        # output_dict = self.context_fn(images_x, images_y)
        loss, LD, LP, advLoss, dexLoss = self.p_losses(images_x,images_y, output_dict, t)
        return loss, self.get_extra_loss(),LD,LP, advLoss, dexLoss
        # return loss

    def scale_to_beta(self, bitrate_scale):
        return 2 ** (3 * bitrate_scale) * 5e-4

