from data import load_data
import argparse
import os
import torch
import torchvision
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import config
from modules.denoising_diffusion_froze_alpha import GaussianDiffusion
from modules.unet import Unet
from modules.compress_modules_froze import BigCompressor, SimpleCompressor
from dataset.PairCityscape import PairCityscape
from dataset.PairKitti import PairKitti
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim
import PIL.Image as Image
from PIL import Image, ImageFilter
import pandas as pd
import cv2
from math import log10
import math
import lpips
from pytorch_fid import fid_score
from lpips import LPIPS
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from DISTS_pytorch import DISTS
from torch.utils.data import DataLoader,SubsetRandomSampler
import time
parser = argparse.ArgumentParser(description="values from bash script")

parser.add_argument("--ckpt", type=str, required=True, default = '') # ckpt path
parser.add_argument("--gamma", type=float, default=0.8) # noise intensity for decoding
parser.add_argument("--n_denoise_step", type=int, default=1) # number of denoising step
parser.add_argument("--device", type=int, default=0) # gpu device index
# parser.add_argument("--img_dir", type=str, default='../imgs')
parser.add_argument("--out_dir", type=str, default='./')
parser.add_argument("--lpips_weight", type=float, required=True,default=0.9) # either 0.9 or 0.0, note that this must match the ckpt you use, because with weight>0, the lpips-vggnet weights were also saved during training. Incorrect state_dict keys may lead to load_state_dict error when loading the ckpt.
args = parser.parse_args()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_MB = trainable_params * 4 / (1024 ** 2)  # float32 -> 4 bytes
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {size_MB:.2f} MB")

def main(rank):

    path = config.data_path
    resize = tuple(config.resize)
    if config.dataset_name == 'KITTI_General' or config.dataset_name == 'KITTI_Stereo':
        stereo = config.dataset_name == 'KITTI_Stereo'
        train_dataset = PairKitti(path=path, set_type='train', stereo=stereo, resize=resize)
        val_dataset = PairKitti(path=path, set_type='val', stereo=stereo, resize=resize)
        test_dataset = PairKitti(path=path, set_type='test', stereo=stereo, resize=resize)
    elif config.dataset_name == 'Cityscape':
        train_dataset = PairCityscape(path=path, set_type='train', resize=resize)
        val_dataset = PairCityscape(path=path, set_type='val', resize=resize)
        test_dataset = PairCityscape(path=path, set_type='test', resize=resize)
    else:
        raise Exception("Dataset not found")
    val_size = len(val_dataset)
    subset_size = val_size//10   # 选取十分之一的数据
    subset_indices = list(range(subset_size))
    val_sampler = SubsetRandomSampler(subset_indices)

    batch_size = config.train_batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=3)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, sampler=val_sampler, num_workers=3)

    denoise_model = Unet(
        dim=64,
        channels=3,
        context_channels=3,
        dim_mults=(1, 2, 3, 4, 5, 6),
        context_dim_mults=(1, 2, 3, 4),
    )

    context_model = BigCompressor(
        dim=64,
        dim_mults=(1, 2, 3, 4),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
        vbr=False,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        num_timesteps=20000,
        loss_type="l1",
        clip_noise="none",
        vbr=False,
        lagrangian=0.9,
        pred_mode="noise",
        var_schedule="linear",
        aux_loss_weight=args.lpips_weight,
        aux_loss_type="lpips"
    )

    loaded_param = torch.load(
        args.ckpt,
        map_location=lambda storage, loc: storage,
    )

    diffusion.load_state_dict(loaded_param["model"],strict=False)
    diffusion.to(rank)
    diffusion.eval()
    # count_parameters(diffusion)
    count_parameters(context_model)
    results_path = os.path.join(args.out_dir, 'kitti_stero_compressed_imgs')
    if not os.path.exists(results_path):
            os.makedirs(results_path)
    names = ["Image Number", "BPP", "PSNR", "MS-SSIM"]
    cols = dict()

    def save_image(x_recon, x, path, name):
        img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
        img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
        img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
        img = np.transpose(img, (1, 2, 0)).astype('uint8')
        # img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
        img_final = Image.fromarray((img_recon),'RGB')        
        if not os.path.exists(path):
            os.makedirs(path)
        img_final.save(os.path.join(path, name + '.png'))
    
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


    def extract_inception_features(img, model):
        """
        Extract features from InceptionV3 model for a single image.

        Parameters:
            img: torch.Tensor, Image of shape (1, 3, H, W), normalized to [0, 1].
            model: torch.nn.Module, Pretrained InceptionV3 model.

        Returns:
            np.ndarray, Feature vector of the image.
        """
        model.eval()
        with torch.no_grad():
            features = model(img)
        return features.cpu().numpy()

    def calculate_fid(features1, features2):
        """
        Calculate FID between two sets of features.

        Parameters:
            features1: np.ndarray, Features of dataset 1, shape (N, D).
            features2: np.ndarray, Features of dataset 2, shape (N, D).

        Returns:
            float, FID value.
        """
        mean1, cov1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mean2, cov2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

        mean_diff = mean1 - mean2
        cov_mean, _ = sqrtm(cov1.dot(cov2), disp=False)

        # Handle numerical instability
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        fid = mean_diff.dot(mean_diff) + np.trace(cov1 + cov2 - 2 * cov_mean)
        return fid

    # Initialize LPIPS and DISTS models
    lpips_model = LPIPS(net='alex').to(config.device)
    dists_model = DISTS().to(config.device)

    # Initialize InceptionV3 model for FID
    inception_model = inception_v3(pretrained=False).to(config.device)
    weights_path = "inception_v3_google-0cc3c7bd.pth"
    inception_model.load_state_dict(torch.load(weights_path))
    inception_model = inception_model.to(config.device)
    inception_model.fc = torch.nn.Identity()  # Remove the final classification layer

    # Metrics dictionary to store results
    # metrics = {'psnr': [], 'ms_ssim': [], 'ms_ssim_db': [], 'bpp': [], 'lpips': [], 'fid': [], 'dists': []}
    metrics = {'psnr': [], 'ms_ssim': [], 'ms_ssim_db': [], 'bpp': [], 'lpips': [], 'dists': [], 'LD':[], 'LP':[]}
    inference_times = []
    with torch.no_grad():
        id = 0
        for i, data in enumerate(iter(test_loader)):
            img, cor_img, _, _ = data
            img = img.to(config.device)
            cor_img = cor_img.to(config.device)
            # if id>0:
            
            loss, aloss, LD, LP, weight, dexloss  = diffusion(img*2.0-1.0,cor_img*2.0-1.0)
            start_time = time.time() 
            compressed, bpp, bpp_y = diffusion.compress(img*2.0-1.0, cor_img*2.0-1.0, 
                                                    sample_steps=args.n_denoise_step,
                                                    sample_mode="ddim",
                                                    bpp_return_mean=False,
                                                    init=torch.randn_like(img) * args.gamma)
            # if id>0
            end_time = time.time()
            # ---- 推理结束 ----
            inference_times.append(end_time - start_time)
            compressed = compressed.clamp(-1, 1) / 2.0 + 0.5


            mse = torch.nn.MSELoss(reduction='mean').to(config.device)
            mse_dist = mse(img, compressed)
            psnr = 20 * log10(1.0 / torch.sqrt(mse_dist))
 
            # Calculate MS-SSIM
            msssim = ms_ssim(img, compressed, data_range=1.0, size_average=True, win_size=7)
            msssim_db = -10 * log10(1 - msssim)

            # Calculate LPIPS
            lpips_score = calculate_lpips(img, compressed)


            # # Calculate DISTS
            dists_score = calculate_dists(img, compressed)

            # Store metrics
            metrics['psnr'].append(psnr)
            metrics['ms_ssim'].append(msssim.item())
            metrics['ms_ssim_db'].append(msssim_db)
            metrics['bpp'].append(bpp.item())
            metrics['lpips'].append(lpips_score)
            # metrics['fid'].append(fid_score)
            metrics['dists'].append(dists_score)
            metrics['LD'].append(LD)
            metrics['LP'].append(LP)            
            print(f"bpp: {bpp.item():.6f}, PSNR: {psnr:.2f}, MS-SSIM: {msssim.item():.4f}, MS-SSIM_DB: {msssim_db:.4f}, LD: {LD:.4f}, LP: {LP:.4f}")
            print(f"LPIPS: {lpips_score:.4f},  DISTS: {dists_score:.4f}")

            id = id+1
        avg_time = sum(inference_times) / len(inference_times)
        print(f"Average inference time per image: {avg_time:.4f} seconds")            
        # Calculate average metrics
        avg_psnr = sum(metrics['psnr']) / len(metrics['psnr'])
        avg_ms_ssim = sum(metrics['ms_ssim']) / len(metrics['ms_ssim'])
        avg_ms_ssim_db = sum(metrics['ms_ssim_db']) / len(metrics['ms_ssim_db'])
        avg_bpp = sum(metrics['bpp']) / len(metrics['bpp'])
        avg_lpips = sum(metrics['lpips']) / len(metrics['lpips'])
        # avg_fid = sum(metrics['fid']) / len(metrics['fid'])
        avg_dists = sum(metrics['dists']) / len(metrics['dists'])

        # Print average metrics for the dataset
        print(f"Average Metrics - bpp: {avg_bpp:.7f}, PSNR: {avg_psnr:.2f}, MS-SSIM: {avg_ms_ssim:.4f}, MS-SSIM_DB: {avg_ms_ssim_db:.4f}")
        print(f"Average LPIPS: {avg_lpips:.4f}, DISTS: {avg_dists:.4f}")


if __name__ == "__main__":
    main(args.device)
