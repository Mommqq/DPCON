import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ms_ssim
from math import log10
import PIL.Image as Image
from PIL import Image, ImageFilter
import os
import math
import numpy as np
import lpips
from collections import OrderedDict
from pytorch_fid import fid_score
from lpips import LPIPS
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from DISTS_pytorch import DISTS

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def calculate_dists(img1, img2):
        """
        Calculate DISTS between two images.

        Parameters:
            img1: torch.Tensor, Image 1 of shape (1, 3, H, W), normalized to [0, 1].
            img2: torch.Tensor, Image 2 of shape (1, 3, H, W), normalized to [0, 1].

        Returns:
            float, DISTS value.
        """
        dists_model = DISTS().to(img1.device)
        dists_value = dists_model(img1, img2)
        return dists_value.item()

def calculate_lpips(img1, img2, model_type='vgg'):
        """
        Calculate LPIPS between two images.

        Parameters:
            img1: torch.Tensor, Image 1 of shape (1, 3, H, W), normalized to [0, 1].
            img2: torch.Tensor, Image 2 of shape (1, 3, H, W), normalized to [0, 1].
            model_type: str, Backbone network for LPIPS ('alex', 'vgg', 'squeeze').

        Returns:
            float, LPIPS value.
        """
        lpips_model = lpips.LPIPS(net=model_type)
        lpips_model = lpips_model.to(img1.device)
        lpips_value = lpips_model(img1, img2)
        return lpips_value.item()

class weight_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化可学习的 log 方差参数（避免除以零）
        self.log_var_dist = nn.Parameter(torch.zeros(1))  # 失真损失权重
        self.log_var_perc = nn.Parameter(torch.zeros(1))  # 感知损失权重
        # self.log_var_adv = nn.Parameter(torch.zeros(1))   # 对抗损失权重

    def forward(self, loss_dist, loss_perc):
        # 计算各损失的权重（与 log 方差成反比）
        weight_dist = 1 / (2 * torch.exp(self.log_var_dist))
        weight_perc = 1 / (2 * torch.exp(self.log_var_perc))
        # weight_adv = 1 / (2 * torch.exp(self.log_var_adv))
        
        # 总损失
        total_loss = (
            weight_dist * loss_dist + 
            weight_perc * loss_perc + 
            # weight_adv * loss_adv +
            self.log_var_dist + 
            self.log_var_perc 
            # self.log_var_adv
        )
        return total_loss

# trainer class
class Trainer(object):
    def __init__(
        self,
        rank,
        sample_steps,
        diffusion_model,
        train_dl,
        val_dl,
        scheduler_function,
        num_epochs = 500000000,
        ema_decay=0.995,
        train_lr=1e-4,
        train_num_steps=1000000,
        scheduler_checkpoint_step=100000,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        tensorboard_dir="./tensorboard_logs/diffusion-video/",
        model_name="model",
        val_num_of_batch=1,
        optimizer="adam",
        sample_mode="ddpm",
        log_file_train = "./kitti_result_distribute_cbam/train_log_alpha0.9_beta0.0128_cbam",
        log_file_val = "./kitti_result_distribute_cbam/val_log_alpha0.9_beta0.0128_cbam"
    ):
        super().__init__()
        self.model = diffusion_model
        # self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.sample_mode = sample_mode
        # self.update_ema_every = update_ema_every
        self.val_num_of_batch = val_num_of_batch
        self.sample_steps = sample_steps

        # self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.train_num_steps = train_num_steps
        self.num_epochs = num_epochs
        self.weight_loss = weight_Loss()
        # self.train_dl_class = train_dl
        # self.val_dl_class = val_dl
        self.train_dl = train_dl
        self.val_dl = val_dl
        if optimizer == "adam":
            self.opt = Adam(self.model.parameters(), lr=train_lr)
        elif optimizer == "adamw":
            self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = LambdaLR(self.opt, lr_lambda=scheduler_function)

        self.step = 0
        self.device = rank
        self.scheduler_checkpoint_step = scheduler_checkpoint_step

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.model_name = model_name
 
        # if os.path.isdir(tensorboard_dir):
        #     shutil.rmtree(tensorboard_dir)
        self.writer = SummaryWriter(tensorboard_dir)
        self.train_log_file = log_file_train
        self.val_log_file = log_file_val

        self.initialize_log_file()
        # self.reset_parameters()

    # def reset_parameters(self):
    #     # self.ema_model.load_state_dict(self.model.state_dict())
    #     pass

    # def step_ema(self):
    #     # if self.step < self.step_start_ema:
    #     #     self.reset_parameters()
    #     # else:
    #     #     self.ema.update_model_average(self.ema_model, self.model)
    #     pass

    def save(self):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            # "ema": self.ema_model.module.state_dict(),
        }
        idx = (self.step // self.save_and_sample_every) % 2
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}_froze_alpha_sigle.pt"))

    def load(self, idx=1, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}_froze_LP.pt"),
            map_location=lambda storage, loc: storage,
        )
        
        if load_step:
            self.step = data["step"]
        try:
                # 获取 checkpoint 中保存的模型参数和当前模型的参数
            checkpoint_state = data["model"]
            model_state = self.model.module.state_dict()

            # 过滤出形状匹配的参数
            filtered_state = {}
            for key, value in checkpoint_state.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        print(f"跳过参数 {key}: checkpoint 中的形状 {value.shape} 与模型中当前的形状 {model_state[key].shape} 不匹配。")
                else:
                    print(f"参数 {key} 不在当前模型中。")
            
            # 更新当前模型的 state_dict
            model_state.update(filtered_state)
            # 加载更新后的 state_dict（由于 filtered_state 已排除不匹配部分，因此这里可以不使用 strict=True）
            self.model.module.load_state_dict(model_state, strict=False)
            # self.model.module.load_state_dict(data["model"], strict=False)
            for name, param in self.model.named_parameters():
                # 冻结 denoise_fn.downs 部分
                if "denoise_fn" in name:
                    param.requires_grad = False
                if "context_fn.dec_x" in name:
                    param.requires_grad = False
                if "context_fn.dec_Px" in name:
                    param.requires_grad = False
                if "context_fn.dec_Py" in name:
                    param.requires_grad = False                                      
                if "context_fn.enc_x" in name:
                    param.requires_grad = False
                if "context_fn.enc_y" in name:
                    param.requires_grad = False
                if "context_fn.prior_x" in name:
                    param.requires_grad = False   
                if "context_fn.prior_y" in name:
                    param.requires_grad = False 
                if "context_fn.dec_y" in name:
                    param.requires_grad = False 
                if "context_fn.dec_xy" in name:
                    param.requires_grad = False 
                if "context_fn.hyper_enc_x" in name:
                    param.requires_grad = False 
                if "context_fn.hyper_enc_y" in name:
                    param.requires_grad = False      
                if "context_fn.hyper_dec_x" in name:
                    param.requires_grad = False 
                if "context_fn.hyper_dec_y" in name:
                    param.requires_grad = False  
                       
            for name, param in self.model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")
        except:
                # 获取 checkpoint 中保存的模型参数和当前模型的参数
            checkpoint_state = data["model"]
            model_state = self.model.state_dict()

            # 过滤出形状匹配的参数
            filtered_state = {}
            for key, value in checkpoint_state.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        print(f"跳过参数 {key}: checkpoint 中的形状 {value.shape} 与模型中当前的形状 {model_state[key].shape} 不匹配。")
                else:
                    print(f"参数 {key} 不在当前模型中。")
            
            # 更新当前模型的 state_dict
            model_state.update(filtered_state)
            # 加载更新后的 state_dict（由于 filtered_state 已排除不匹配部分，因此这里可以不使用 strict=True）
            self.model.load_state_dict(model_state, strict=False)
            # self.model.load_state_dict(data["model"], strict=False)
            for name, param in self.model.named_parameters():
                # 冻结 denoise_fn.downs 部分
                if "denoise_fn" in name:
                    param.requires_grad = False
                if "context_fn.dec_x." in name:
                    param.requires_grad = False
                if "context_fn.dec_Px" in name:
                    param.requires_grad = False
                if "context_fn.dec_Py" in name:
                    param.requires_grad = False
                if "context_fn.enc_x" in name:
                    param.requires_grad = False
                if "context_fn.enc_y" in name:
                    param.requires_grad = False
                if "context_fn.prior_x" in name:
                    param.requires_grad = False   
                if "context_fn.prior_y" in name:
                    param.requires_grad = False 
                if "context_fn.dec_y" in name:
                    param.requires_grad = False 
                if "context_fn.dec_xy" in name:
                    param.requires_grad = False 
                if "context_fn.hyper_enc_x" in name:
                    param.requires_grad = False 
                if "context_fn.hyper_enc_y" in name:
                    param.requires_grad = False      
                if "context_fn.hyper_dec_x" in name:
                    param.requires_grad = False 
                if "context_fn.hyper_dec_y" in name:
                    param.requires_grad = False  
                       
            for name, param in self.model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")  

    def initialize_log_file(self):
        """初始化日志文件（如果文件不存在，则创建）"""
        # if not os.path.exists(self.train_log_file):
        #     with open(self.train_log_file, 'a') as f:
        #         f.write("Epoch, Avg aloss, Avg loss\n")  # 写入表头
        if not os.path.exists(self.val_log_file):
            with open(self.val_log_file, 'a') as f:
                f.write("Epoch, Avg PSNR, Avg ms-ssim, Avg Loss, Avg Bpp\n")  # 写入表头
        # for name, param in self.model.named_parameters():

                # if "denoise_fn" in name:
                #     param.requires_grad = False
                
                # if "context_fn.dec_x." in name:
                #     param.requires_grad = False
                     
        #         if "context_fn.ca" in name:
        #             param.requires_grad = False                 
   
        
        # for name, param in self.model.named_parameters():
        #         print(f"{name}: requires_grad={param.requires_grad}")        

    def val_log_to_file(self, epoch, avg_psnr, avg_ms_ssim, avg_loss, avg_bpp, avg_bpp_y,avg_LD,avg_LP,avg_LPIPS):
        """将验证指标写入日志文件"""
        with open(self.val_log_file, 'a') as f: 
            f.write(f"{epoch}, {avg_psnr:.2f}, {avg_ms_ssim:.4f}, {avg_loss:.4f}, {avg_bpp: .4f}, {avg_bpp_y: .4f}, {avg_LD: .4f}, {avg_LP: .4f}, {avg_LPIPS: .4f}\n")
    
    # def train_log_to_file(self, epoch, avg_loss):
    #     """将训练指标写入日志文件"""
    #     with open(self.train_log_file, 'a') as f:  
    #         # f.write(f"{epoch}, {avg_loss:.2f}, {avg_aloss:.4f}\n")
    #         f.write(f"{epoch}, {avg_loss:.2f}")
    def save_image(self, x_recon, x, path, name):
        img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
        img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
        img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
        img = np.transpose(img, (1, 2, 0)).astype('uint8')
        # img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
        img_final = Image.fromarray((img_recon),'RGB')        
        if not os.path.exists(path):
            os.makedirs(path)
        img_final.save(os.path.join(path, name + '.png'))
    def train(self):

        # total_epochs = self.train_num_steps // len(self.train_dl)
        plt.ion() 
        for epoch in range(self.num_epochs):
            total_aloss = []
            total_loss = []
            total_LD = []
            total_LP = []
      
            print(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            pbar = tqdm(total=len(self.train_dl), desc=f"Epoch {epoch + 1}")
            for data_x,data_y,_,_ in self.train_dl:
                # P_weight = 0.1 + (0.9 - 0.1) * 0.5 * (1 - math.cos(math.pi * (self.step) / self.train_num_steps)) if self.step<self.train_num_steps else 0.9
                P_weight = self.model.alpha
                # P_weight = P_weight.detach()
                
                self.opt.zero_grad()
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                self.model.train()
                loss, aloss, LD, LP, advLoss, dexLoss = self.model(data_x * 2.0 - 1.0,data_y * 2.0 - 1.0)
                # if self.step%20000 == 0:
                #     compressed, bpp, bpp_y = self.model.compress(data_x * 2.0 - 1.0, data_y * 2.0 - 1.0, self.sample_steps, None, self.sample_mode)
                #     compressed = (compressed.clamp(-1, 1) + 1.0) * 0.5       
                #     self.save_image(compressed[0], data_x[0], os.path.join('sigle_city', 'alpha0.0128_images_700'), str(self.step))
                #     self.save_image(data_x[0], compressed[0], os.path.join('sigle_city', 'alpha0.0128_images_700'), 'orange'+str(self.step))
                # loss = loss+P_weight*LD+(1-P_weight)*LP+aloss   
                # loss = loss+LD+aloss 
                LP_total = LP + dexLoss  

                # if self.step % 10 == 0:  # 每 100 轮更新一次热力图
                #     # print(torch.sigmoid(self.model.alpha_base))  
                    
                #     k = torch.sigmoid(self.model.alpha_base.detach().cpu()) # 取出数值，不影响梯度计算
                #     heatmap_data = k.mean(dim=1).squeeze()  # 取均值，得到 (128, 256)

                #     plt.clf()  # 清除上一次绘图
                #     sns.heatmap(heatmap_data.numpy(), cmap='viridis', cbar=True)  
                #     plt.title(f"alpha Heatmap - Step {self.step}")
                #     # plt.savefig("city_weight_100000.png") if self.step  == 60000 else plt.savefig("city_weight_train.png")
                #     plt.savefig("sigle_city/alpha0.0128_images_700/weight_train_sigle_700.png")
                #     plt.pause(0.1)  # 暂停 0.1 秒，让图像更新    
                loss.backward()
                
                # LP_total.backward()
                # max_loss = max(LD.item(), LP.item())
                # alpha_loss =  (1 + torch.log(1 + LP / LD)) * LD + (1 + torch.log(1 + LD / LP)) * LP 
                # alpha_loss = (LP/max_loss)*LD+(LD/max_loss)*LP
                # self.weight_loss.to(self.device)
                # alpha_loss = self.weight_loss(LD,LP)
                # alpha_loss.backward()  
                
                self.opt.step()
                total_loss.append(loss.item())
                total_aloss.append(aloss.item())
                total_LD.append(LD.item()) 
                total_LP.append(LP.item())              
                pbar.update(1)  # 更新进度条
                
                if self.step % self.scheduler_checkpoint_step == 0 and self.step != 0:
                    self.scheduler.step()
                
                self.step += 1
              
            # print("alpha_loss:",alpha_loss.item())
            avg_aloss = sum(total_aloss)/len(total_aloss)
            avg_loss = sum(total_loss)/len(total_loss)
            print("avg_loss:",avg_loss," adv_loss:",advLoss.item(),"LD:",sum(total_LD)/len(total_LD), "LP:",sum(total_LP)/len(total_LP),"P_weight:",P_weight, 'lr:',self.opt.param_groups[0]['lr'])
            # print(" avg_loss:",avg_loss)
            # print("alpha:",self.model.aux_loss_weight)
            # self.train_log_to_file(epoch + 1, avg_loss, avg_aloss)
            # self.train_log_to_file(epoch + 1, avg_loss)
            pbar.close()
            print("current step:",self.step)
            self.save()            
            if (epoch + 1) % 1500 == 0: 
                # print(torch.sigmoid(self.model.alpha_base))
                metrics = self.validate()
                print(f"Epoch {epoch + 1}: Validation Metrics: {metrics}")
                self.val_log_to_file(epoch + 1, metrics['avg_psnr'], metrics['avg_ms_ssim'], metrics['avg_loss'], metrics['avg_bpp'], metrics['avg_bpp_y'], metrics['avg_LD'], metrics['avg_LP'],metrics['avg_LPIPS'])

            idx = (self.step // self.save_and_sample_every) % 2
            print("epoch",epoch+1,"idx:",idx)
        self.save()
        print("Training completed")

    def validate(self):
        """执行验证并收集指标,包括PSNR和MS-SSIM"""
        self.model.eval()
        metrics = {'psnr': [], 'ms_ssim': [], 'loss':[], 'bpp':[], 'bpp_y':[], 'LD':[], 'LP':[],'LPIPS':[]}
        lpips_ = []
        # avg_lpips = []
        # LP_ = []
        # LD_ = []
        total_loss = 0
        total_bpp = 0
        num_batch = 0
        with torch.no_grad():
            for batch_x,batch_y,_,_ in self.val_dl:
                # if i >= self.val_num_of_batch:
                #     break
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                data_x = batch_x * 2.0 - 1.0
                data_y = batch_y * 2.0 - 1.0
                compressed, bpp, bpp_y = self.model.compress(data_x, data_y, self.sample_steps, None, self.sample_mode)
                compressed = (compressed.clamp(-1, 1) + 1.0) * 0.5 
                for i in range(batch_x.size(0)):  
                    x = batch_x[i].unsqueeze(0) 
                    y = compressed[i].unsqueeze(0) 

                    # 计算LPIPS
                    lpips_score = calculate_lpips(x, y) 
                    lpips_.append(lpips_score)  

                average_lpips = sum(lpips_) / len(lpips_)
     
                # lpips_score = calculate_lpips(batch_x, compressed)
                # lpips_.append(lpips_score)

                loss,aloss,LD,LP, advLoss, dexLoss  = self.model(data_x,data_y)
                loss = loss+LD+LP+aloss
                total_loss += loss.item()
       
                # 计算PSNR
                mse = torch.mean((batch_x - compressed) ** 2)
                psnr = 20 * log10(1.0 / torch.sqrt(mse))
                metrics['psnr'].append(psnr)
                metrics['bpp'].append(bpp)
                metrics['bpp_y'].append(bpp_y)
                # metrics['bpp_w'].append(bpp_w)               
                metrics['loss'].append(loss)
                metrics['LPIPS'].append(average_lpips)
                metrics['LD'].append(LD)
                metrics['LP'].append(LP)
                # 计算MS-SSIM
                ms_ssim_value = ms_ssim(batch_x, compressed, data_range=1.0, size_average=True, win_size=7)
                metrics['ms_ssim'].append(ms_ssim_value.item()) 
                # num_batch+=1
                # total_bpp+=bpp.item()
                print("bpp:",bpp.item(),"psnr:",psnr,"ms-ssim:",ms_ssim_value.item(),"advLoss:",advLoss.item(),"dexLoss:",dexLoss.item())
        # 计算平均指标:
        # print("LD:",sum(LD_)/len(LD_),"LP",sum(LP_)/len(LP_),"LPIPS:",sum(avg_lpips)/len(avg_lpips))
        avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
        avg_ms_ssim = sum(metrics['ms_ssim']) / len(metrics['ms_ssim'])
        avg_bpp = sum(metrics['bpp']) / len(metrics['bpp'])
        avg_bpp_y = sum(metrics['bpp_y']) / len(metrics['bpp_y']) 
        # avg_bpp_w = sum(metrics['bpp_w']) / len(metrics['bpp_w'])  
        avg_loss= sum(metrics['loss']) / len(metrics['loss'])
        avg_LD= sum(metrics['LD']) / len(metrics['LD'])  
        avg_LP= sum(metrics['LP']) / len(metrics['LP'])  
        avg_LPIPS= sum(metrics['LPIPS']) / len(metrics['LPIPS'])      

        print(f"avg bpp: {avg_bpp:.4f}, avg bpp_y: {avg_bpp_y:.4f},  Avg PSNR: {avg_psnr:.2f}, "
      f"Avg MS-SSIM: {avg_ms_ssim:.4f}, Avg Loss: {avg_loss:.4f},  Avg LD: {avg_LD:.4f}, "
      f"Avg LP: {avg_LP:.4f},  Avg LPIPS: {avg_LPIPS:.4f}")

        return {
                    'avg_bpp': avg_bpp,
                    'avg_bpp_y': avg_bpp_y,
                    'avg_psnr': avg_psnr,
                    'avg_ms_ssim': avg_ms_ssim,
                    'avg_loss': avg_loss,
                    'avg_LD': avg_LD,
                    'avg_LP': avg_LP,
                    'avg_LPIPS': avg_LPIPS
                }

    