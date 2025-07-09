import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage import io

def compute_avg_ssim(dataset_a_dir, dataset_b_dir):
    # 获取数据集A中的所有文件名
    filenames = [f for f in os.listdir(dataset_a_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_ssim = 0.0
    count = 0

    for filename in filenames:
        # 构建文件路径
        img_a_path = os.path.join(dataset_a_dir, filename)
        img_b_path = os.path.join(dataset_b_dir, filename)

        # 读取图像
        img_a = io.imread(img_a_path)
        img_b = io.imread(img_b_path)

        if img_a.shape != (128, 256, 3) or img_b.shape != (128, 256, 3):
            print(f"跳过 {filename}: 尺寸不匹配")
            continue

        # 计算SSIM（假设图像为uint8，范围0-255）
        # 使用channel_axis处理多通道，适用于skimage >= 0.19
        try:
            current_ssim = ssim(img_a, img_b, data_range=255, channel_axis=-1)
        except TypeError:
            # 兼容旧版本skimage（multichannel参数）
            current_ssim = ssim(img_a, img_b, data_range=255, multichannel=True)

        total_ssim += current_ssim
        count += 1

    if count == 0:
        raise ValueError("没有有效的图像对进行计算")
    return total_ssim / count

# 使用示例
dataset_a_path = "kitti_stero_compressed_imgs/orange_alpha1.0_"
dataset_b_path = "outputs_ATN/results/kitti_0.0001"
average_ssim = compute_avg_ssim(dataset_a_path, dataset_b_path)
print(f"平均SSIM: {average_ssim:.4f}")