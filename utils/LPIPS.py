import lpips
import torch
from torchvision import transforms
from PIL import Image
import os

# 初始化LPIPS模型（默认使用AlexNet骨干）
loss_fn = lpips.LPIPS(net='vgg').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='vgg')

# 图像预处理
preprocess = transforms.Compose([
    # transforms.Resize((128, 256)),     # 调整到模型期望的尺寸
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
])

def calculate_avg_lpips(dir1, dir2):
    # 获取排序后的文件列表（假设文件名一一对应）
    files1 = sorted([f for f in os.listdir(dir1) if f.endswith(('.png', '.jpg'))])
    files2 = sorted([f for f in os.listdir(dir2) if f.endswith(('.png', '.jpg'))])
    
    assert len(files1) == len(files2), "数据集数量不匹配"
    
    total = 0.0
    for f1, f2 in zip(files1, files2):
        # 加载图像对
        img1 = preprocess(Image.open(os.path.join(dir1, f1)).convert('RGB'))
        img2 = preprocess(Image.open(os.path.join(dir2, f2)).convert('RGB'))
        
        # 转换为批处理格式并送显存
        if torch.cuda.is_available():
            img1 = img1.unsqueeze(0).cuda()
            img2 = img2.unsqueeze(0).cuda()
        
        # 计算相似度
        with torch.no_grad():
            dist = loss_fn(img1, img2)
            total += dist.item()
    
    return total / len(files1)

# 使用示例
dataset_a = "kitti_stero_compressed_imgs/orange_alpha1.0_"
dataset_b = "kitti_stero_compressed_imgs/fusion_alpha0.0064_imagescity_test_alpha1.0_"
avg_lpips = calculate_avg_lpips(dataset_a, dataset_b)
print(f"Average LPIPS: {avg_lpips:.4f}")