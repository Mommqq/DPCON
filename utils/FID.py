import torch
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm
import os
from tqdm import tqdm  

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载Inception-v3模型
model = inception_v3(pretrained=True, transform_input=False)
model.fc = torch.nn.Identity() 
model.eval()

# 提取特征
def extract_features(dataloader, model):
    features = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting Features"):  # 添加进度条
            if torch.cuda.is_available():
                images = images.cuda()
                model = model.cuda()
            outputs = model(images)
            features.append(outputs.cpu())
    return torch.cat(features, dim=0)

# 计算FID
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# 数据集路径
real_img_dir = " "  # 替换为真实图像路径
fake_img_dir = " "  # 替换为生成图像路径

# 加载数据集
real_dataset = ImageDataset(real_img_dir, transform=transform)
fake_dataset = ImageDataset(fake_img_dir, transform=transform)

real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# 提取特征
real_features = extract_features(real_loader, model)
fake_features = extract_features(fake_loader, model)

# 计算FID
fid_value = calculate_fid(real_features.numpy(), fake_features.numpy())
print(f"FID: {fid_value}")
