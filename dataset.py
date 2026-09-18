# -*- coding: utf-8 -*-
"""
dataset.py - 数据集模块

包含 FullDataset 类，用于加载医学图像分割数据。
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FullDataset(Dataset):
    """医学图像分割数据集

    Args:
        image_dir (str): 图像目录路径
        mask_dir (str): 掩码目录路径
        img_size (int): 图像尺寸（长和宽），默认 256
        mode (str): 模式，'val' 或 'test'，默认 'val'
        in_channels (int): 输入通道数，默认 1（灰度图）
    """

    def __init__(self, image_dir, mask_dir, img_size=256, mode='val', in_channels=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        self.in_channels = in_channels

        # 收集所有图像文件
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"在 {image_dir} 中未找到图像文件")

        print(f"FullDataset: 找到 {len(self.image_files)} 张图像 (模式: {mode}, 输入通道: {in_channels})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 读取图像
        if self.in_channels == 1:
            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            original_h, original_w = original_img.shape
            image = Image.open(img_path).convert('L')
        else:
            original_img = cv2.imread(img_path)
            original_h, original_w = original_img.shape[:2]
            image = Image.open(img_path).convert('RGB')

        # 查找对应的掩码
        base_name = os.path.splitext(img_name)[0]
        mask_name = None
        for ext in ['.png', '.jpg', '.bmp', '.tif', '.tiff']:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_name = candidate
                break
            candidate = os.path.join(self.mask_dir, base_name + '_mask' + ext)
            if os.path.exists(candidate):
                mask_name = candidate
                break
            candidate = os.path.join(self.mask_dir, base_name + '_label' + ext)
            if os.path.exists(candidate):
                mask_name = candidate
                break

        if mask_name:
            mask = Image.open(mask_name).convert('L')
        else:
            mask = Image.new('L', (original_w, original_h), 0)

        # Resize 到指定尺寸
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 转 numpy
        image_np = np.array(image, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)

        # 归一化图像
        if self.in_channels == 1:
            mean = image_np.mean()
            std = image_np.std()
            image_np = (image_np - mean) / (std + 1e-8)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W]
        else:
            image_np = image_np.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
            image_tensor = torch.from_numpy(image_np).float()

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        return {
            'image': image_tensor,
            'label': mask_tensor,
            'case': base_name,
            'image_path': img_path,
            'mask_path': mask_name if mask_name else '',
            'original_size': (original_h, original_w)
        }
