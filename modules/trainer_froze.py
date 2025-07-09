import torch
from pathlib import Path
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import cycle
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ms_ssim
from math import log10
import os
import math

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
        num_epochs = 3000,
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
        torch.save(data, str(self.results_folder / f"{self.model_name}_{idx}.pt"))

    def load(self, idx=0, load_step=True):
        data = torch.load(
            str(self.results_folder / f"{self.model_name}_{idx}.pt"),
            map_location=lambda storage, loc: storage,
        )

        if load_step:
            self.step = data["step"]
        try:
            self.model.module.load_state_dict(data["model"], strict=False)
            for name, param in self.model.named_parameters():
                # 冻结 denoise_fn.downs 部分
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
            self.model.load_state_dict(data["model"], strict=False)
   

    def initialize_log_file(self):
        """初始化日志文件（如果文件不存在，则创建）"""
        if not os.path.exists(self.train_log_file):
            with open(self.train_log_file, 'a') as f:
                f.write("Epoch, Avg aloss, Avg loss\n")  # 写入表头
        if not os.path.exists(self.val_log_file):
            with open(self.val_log_file, 'a') as f:
                f.write("Epoch, Avg PSNR, Avg ms-ssim, Avg Loss, Avg Bpp\n")  # 写入表头
        # for name, param in self.model.named_parameters():
        #         # 冻结 denoise_fn.downs 部分
        #         if "denoise_fn" in name:
        #             param.requires_grad = False
        #         # 可以添加其他条件来冻结更多层
        #         if "context_fn.dec_y_cond" in name:
        #             param.requires_grad = False   
        #         if "context_fn.dec_x_cond" in name:
        #             param.requires_grad = False 
        #         if "context_fn.dec_xy_cond" in name:
        #             param.requires_grad = False 
        
        # for name, param in self.model.named_parameters():
        #         print(f"{name}: requires_grad={param.requires_grad}")        

    def val_log_to_file(self, epoch, avg_psnr, avg_ms_ssim, avg_loss, avg_bpp, avg_bpp_y):
        """将验证指标写入日志文件"""
        with open(self.val_log_file, 'a') as f:  # 以追加模式打开文件
            f.write(f"{epoch}, {avg_psnr:.2f}, {avg_ms_ssim:.4f}, {avg_loss:.4f}, {avg_bpp: .4f}, {avg_bpp_y: .4f}\n")
    
    def train_log_to_file(self, epoch, avg_loss):
        """将训练指标写入日志文件"""
        with open(self.train_log_file, 'a') as f:  # 以追加模式打开文件
            # f.write(f"{epoch}, {avg_loss:.2f}, {avg_aloss:.4f}\n")
            f.write(f"{epoch}, {avg_loss:.2f}")

    def train(self):

        # total_epochs = self.train_num_steps // len(self.train_dl)
        for epoch in range(self.num_epochs):
            total_aloss = []
            total_loss = []
            print(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            pbar = tqdm(total=len(self.train_dl), desc=f"Epoch {epoch + 1}")
            for data_x,data_y,_,_ in self.train_dl:
                # self.model.aux_loss_weight = 0.1 + (0.9 - 0.1) * 0.5 * (1 - math.cos(math.pi * (self.step) / self.train_num_steps))
                self.opt.zero_grad()
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                self.model.train()
                loss, aloss = self.model(data_x * 2.0 - 1.0,data_y * 2.0 - 1.0)
                # loss = self.model(data_x * 2.0 - 1.0,data_y * 2.0 - 1.0)
                loss.backward()
                aloss.backward()
                self.opt.step()
                total_loss.append(loss.item())
                total_aloss.append(aloss.item())
                pbar.update(1)  # 更新进度条
                
                if self.step % self.scheduler_checkpoint_step == 0 and self.step != 0:
                    self.scheduler.step()
                
                self.step += 1
            avg_aloss = sum(total_aloss)/len(total_aloss)
            avg_loss = sum(total_loss)/len(total_loss)
            print("avg_aloss:",avg_aloss," avg_loss:",avg_loss)
            # print(" avg_loss:",avg_loss)
            # print("alpha:",self.model.aux_loss_weight)
            # self.train_log_to_file(epoch + 1, avg_loss, avg_aloss)
            self.train_log_to_file(epoch + 1, avg_loss)
            pbar.close()
            print("current step:",self.step)
            self.save()            
            if (epoch + 1) % 10 == 0:  
                metrics = self.validate()
                print(f"Epoch {epoch + 1}: Validation Metrics: {metrics}")
                self.val_log_to_file(epoch + 1, metrics['avg_psnr'], metrics['avg_ms_ssim'], metrics['avg_loss'], metrics['avg_bpp'], metrics['avg_bpp_y'])

            idx = (self.step // self.save_and_sample_every) % 2
            print("epoch",epoch+1,"idx:",idx)
        self.save()
        print("Training completed")

    def validate(self):
        self.model.eval()
        metrics = {'psnr': [], 'ms_ssim': [], 'loss':[], 'bpp':[], 'bpp_y':[], 'loss':[]}
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

                # 计算损失
                loss,aloss  = self.model(data_x,data_y)
                total_loss += loss.item()

                # 计算PSNR
                mse = torch.mean((batch_x - compressed) ** 2)
                psnr = 20 * log10(1.0 / torch.sqrt(mse))
                metrics['psnr'].append(psnr)
                metrics['bpp'].append(bpp)
                metrics['bpp_y'].append(bpp_y)
                # metrics['bpp_w'].append(bpp_w)               
                metrics['loss'].append(loss)

                # 计算MS-SSIM
                ms_ssim_value = ms_ssim(batch_x, compressed, data_range=1.0, size_average=True, win_size=7)
                metrics['ms_ssim'].append(ms_ssim_value.item()) 
                # num_batch+=1
                # total_bpp+=bpp.item()
                print("bpp:",bpp.item(),"psnr:",psnr,"ms-ssim:",ms_ssim_value.item())
        # 计算平均指标
        avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
        avg_ms_ssim = sum(metrics['ms_ssim']) / len(metrics['ms_ssim'])
        avg_bpp = sum(metrics['bpp']) / len(metrics['bpp'])
        avg_bpp_y = sum(metrics['bpp_y']) / len(metrics['bpp_y']) 
        # avg_bpp_w = sum(metrics['bpp_w']) / len(metrics['bpp_w'])  
        avg_loss= sum(metrics['loss']) / len(metrics['loss'])    
        print(f"avg bpp: {avg_bpp:.4f}, avg bpp_y: {avg_bpp_y:.4f},  Avg PSNR: {avg_psnr:.2f}, Avg MS-SSIM: {avg_ms_ssim:.4f}, Avg Loss: {avg_loss:.4f}")
        return {'avg_bpp': avg_bpp, 'avg_bpp_y': avg_bpp_y,  'avg_psnr': avg_psnr, 'avg_ms_ssim': avg_ms_ssim, 'avg_loss': avg_loss}
    