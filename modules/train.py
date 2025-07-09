import torch
from data import load_data
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
# from modules.denoising_diffusion_dualdecoder import GaussianDiffusion
from modules.denoising_diffusion_froze import GaussianDiffusion
# from modules.denoising_diffusion_froze_alpha import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
# from modules.compress_modules_dualdecoder import BigCompressor, SimpleCompressor
from modules.compress_modules_froze import BigCompressor, SimpleCompressor
import config
from dataset.PairCityscape import PairCityscape
from dataset.PairKitti import PairKitti
from torch.utils.data import DataLoader,SubsetRandomSampler


model_name = (
    f"{config.compressor}-{config.loss_type}-{config.dataset_name}"
    f"-d{config.embed_dim}-t{config.iteration_step}-b{config.beta}-vbr{config.vbr}"
    f"-{config.pred_mode}-{config.var_schedule}-aux{config.alpha}{config.aux_loss_type if config.alpha>0 else ''}{config.additional_note}"
)


def schedule_func(ep):
    return max(config.decay ** ep, config.minf)


def main():

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

    train_size = len(train_dataset)
    subset_size1 = train_size//1576   # 选取十分之一的数据
    # subset_indices1 = list(range(subset_size1))
    subset_indices1 = [700]
    train_sampler = SubsetRandomSampler(subset_indices1)

    batch_size = config.train_batch_size

   
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=3)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=3)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=3)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, sampler=val_sampler, num_workers=3)
    
    denoise_model = Unet(
        dim=config.embed_dim,
        channels=config.data_config["img_channel"],
        context_channels=config.context_channels,
        dim_mults=config.dim_mults,
        context_dim_mults=config.context_dim_mults
    )

    if config.compressor == 'big':
        context_model = BigCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=config.context_channels,
            vbr=config.vbr
        )
    elif config.compressor == 'simple':
        context_model = SimpleCompressor(
            dim=config.embed_dim,
            dim_mults=config.context_dim_mults,
            hyper_dims_mults=config.hyper_dim_mults,
            channels=config.data_config["img_channel"],
            out_channels=config.context_channels,
            vbr=config.vbr
        )
    # elif config.compressor == 'ATN':
    #     context_model = CADistributedAutoEncoder()   
    else:
        raise NotImplementedError

    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        context_fn=context_model,
        clip_noise=config.clip_noise,
        num_timesteps=config.iteration_step,
        loss_type=config.loss_type,
        vbr=config.vbr,
        lagrangian=config.beta,
        pred_mode=config.pred_mode,
        aux_loss_weight=config.alpha,
        aux_loss_type=config.aux_loss_type,
        var_schedule=config.var_schedule
    ).to(config.device)

    trainer = Trainer(
        rank=config.device,
        sample_steps=config.sample_steps,
        diffusion_model=diffusion,
        train_dl=train_loader,
        val_dl=val_loader,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{model_name}/"),
        model_name=model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer,
        sample_mode=config.sample_mode,
        log_file_train = config.train_path,
        log_file_val = config.val_path
    )

    if config.load_model:
        trainer.load(load_step=config.load_step)

    trainer.train()


if __name__ == "__main__":
    main()

