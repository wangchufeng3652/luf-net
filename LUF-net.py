import os
import json
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
import math

# -----------------------------
# 路径配置（根据实际修改）
# -----------------------------
inputs_dir = “input_path”
targets_dir = “target_path”
vector_json = “vectors.json_path"
best_model_path = "best_model.pth_path"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# -----------------------------
# RGB <-> Lab 转换函数
# -----------------------------
def rgb_to_lab(img_tensor):
    if img_tensor.dim() == 3:
        img = (img_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab = torch.tensor(lab, dtype=torch.float32).permute(2,0,1)/255.0
        return lab.to(img_tensor.device)
    elif img_tensor.dim() == 4:
        return torch.stack([rgb_to_lab(img_tensor[i]) for i in range(img_tensor.size(0))])

def lab_to_rgb(lab_tensor):
    if lab_tensor.dim() == 3:
        lab = (lab_tensor*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return torch.tensor(rgb, dtype=torch.float32).permute(2,0,1)/255.0
    elif lab_tensor.dim() == 4:
        return torch.stack([lab_to_rgb(lab_tensor[i]) for i in range(lab_tensor.size(0))])

# -----------------------------
# 加载 maize vectors
# -----------------------------
with open(vector_json, 'r') as f:
    vec_dict = json.load(f)

all_keys = sorted(vec_dict.keys(), key=lambda x: int(x))
all_vectors = []
for k in all_keys:
    v = vec_dict[k]
    vec7 = [v['aperture'], v['exposure_time'], v['iso'],
            v['band_G'], v['band_R'], v['band_NIR'], v['band_RE']]
    all_vectors.append(vec7)
all_vectors = torch.tensor(all_vectors, dtype=torch.float32)

# -----------------------------
# 文件列表
# -----------------------------
input_paths = sorted([os.path.join(inputs_dir, f) for f in os.listdir(inputs_dir) if f.endswith('.jpg')])
target_paths = sorted([os.path.join(targets_dir, f) for f in os.listdir(targets_dir) if f.endswith('.jpg')])

assert len(input_paths) == len(target_paths) == len(all_vectors), "输入、目标与向量数量不一致！"

# -----------------------------
# Dataset
# -----------------------------
class ImageVectorDatasetLab(Dataset):
    def __init__(self, input_paths, target_paths, vectors):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.vectors = vectors.to(device)
    def __len__(self):
        return len(self.input_paths)
    def __getitem__(self, idx):
        inp = read_image(self.input_paths[idx]).float()/255.0
        tgt = read_image(self.target_paths[idx]).float()/255.0
        inp_lab = rgb_to_lab(inp)
        tgt_lab = rgb_to_lab(tgt)
        return inp_lab.to(device), self.vectors[idx], tgt_lab.to(device)

dataset = ImageVectorDatasetLab(input_paths, target_paths, all_vectors)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -----------------------------
# 模型定义（与训练一致）
# -----------------------------
class FiLM(nn.Module):
    def __init__(self, vector_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(vector_dim, feature_dim)
        self.beta = nn.Linear(vector_dim, feature_dim)
    def forward(self, x, vec):
        gamma = self.gamma(vec).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(vec).unsqueeze(-1).unsqueeze(-1)
        return x*(1+gamma)+beta

class LiteUNetFiLMResidualLab(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, vector_dim=7):
        super().__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_channels, 32)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), CBR(32, 64))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), CBR(64, 128))
        self.bottleneck = CBR(128, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = CBR(256+128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = CBR(128+64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        self.film1 = FiLM(vector_dim, 32)
        self.film2 = FiLM(vector_dim, 64)
        self.film3 = FiLM(vector_dim, 128)

    def forward(self, x_img, x_vec):
        e1 = self.film1(self.enc1(x_img), x_vec)
        e2 = self.film2(self.enc2(e1), x_vec)
        e3 = self.film3(self.enc3(e2), x_vec)
        b = self.bottleneck(e3)
        d2 = self.up2(b)
        if d2.shape[2:] != e3.shape[2:]:
            d2 = F.interpolate(d2, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d1 = self.up1(d2)
        if d1.shape[2:] != e2.shape[2:]:
            d1 = F.interpolate(d1, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        residual = self.out_conv(d1)
        residual = F.interpolate(residual, size=x_img.shape[2:], mode='bilinear', align_corners=True)
        out = x_img + residual
        return torch.clamp(out, 0.0, 1.0)

# -----------------------------
# 加载模型并评估
# -----------------------------
model = LiteUNetFiLMResidualLab(vector_dim=7).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

criterion = nn.MSELoss()
color_error, test_mape = 0, 0

torch.cuda.synchronize() if torch.cuda.is_available() else None
start_time = time.time()


with torch.no_grad():
    for imgs, vecs, targets in loader:
        pred = model(imgs, vecs)
        # 确保 pred 和 targets 都在 Lab 空间
        diff = pred - targets  # N x 3 x H x W
        # 每个像素的欧氏距离
        delta_e = torch.sqrt(torch.sum(diff**2, dim=1))  # N x H x W
        # 每张图片的平均 color error
        color_error += delta_e.mean().item() * imgs.size(0)
        mape = torch.mean(torch.abs(pred - targets)/(torch.abs(targets)+1e-6))*100
        test_mape += mape.item() * imgs.size(0)

torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start_time

color_error /= len(loader.dataset)
test_mape /= len(loader.dataset)

print(f"\n🌽 Maize 数据集评估结果：")
print(f"✅ color error (Lab): {color_error*(255):.2f}")
print(f"✅ MAPE: {test_mape:.2f}%")
print(f"⏱️ 总耗时: {elapsed:.2f}s")
print(f"⚡ 每张耗时: {elapsed/len(loader.dataset):.6f}s")
print(f"📊 吞吐率 (FPS): {len(loader.dataset)/elapsed:.2f}")
