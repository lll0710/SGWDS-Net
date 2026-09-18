import os
import sys
import argparse
import time
import datetime
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from SGWDSNet import SGWDSNet, compute_gradient_flow_from_mask




def compute_flow_gt_batch(masks, device='cpu'):
   
    B, _, H, W = masks.shape
    flow_gt = torch.zeros(B, 2, H, W, dtype=torch.float32)

    masks_np = masks.squeeze(1).cpu().numpy()

    for i in range(B):
        mask = masks_np[i]
        if mask.max() > 0:
            flow = compute_gradient_flow_from_mask(mask)
            flow_gt[i] = torch.from_numpy(flow)

    return flow_gt.to(device)



class OnlineAugmentation:
    

    def __init__(self,
                 rotation_range: tuple = (-30, 30),
                 scale_range: tuple = (0.7, 1.3),
                 brightness_range: tuple = (0.7, 1.3),
                 contrast_range: tuple = (0.7, 1.3),
                 gaussian_noise_std: float = 0.05,
                 elastic_alpha: float = 100,
                 elastic_sigma: float = 10,
                 prob_rotation: float = 0.7,
                 prob_scale: float = 0.6,
                 prob_flip: float = 0.5,
                 prob_brightness: float = 0.4,
                 prob_contrast: float = 0.4,
                 prob_noise: float = 0.4,
                 prob_blur: float = 0.3,
                 prob_elastic: float = 0.4,
                 min_augmentations: int = 2,
                 double_aug_prob: float = 0.3):

        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gaussian_noise_std = gaussian_noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma

        self.prob_rotation = prob_rotation
        self.prob_scale = prob_scale
        self.prob_flip = prob_flip
        self.prob_brightness = prob_brightness
        self.prob_contrast = prob_contrast
        self.prob_noise = prob_noise
        self.prob_blur = prob_blur
        self.prob_elastic = prob_elastic

        self.min_augmentations = min_augmentations
        self.double_aug_prob = double_aug_prob

    def __call__(self, image: np.ndarray, mask: np.ndarray):
       
        aug_count = 0

        if random.random() < self.prob_rotation:
            image, mask = self._random_rotation(image, mask)
            aug_count += 1

        if random.random() < self.prob_scale:
            image, mask = self._random_scale(image, mask)
            aug_count += 1

        if random.random() < self.prob_flip:
            image, mask = self._random_flip(image, mask)
            aug_count += 1

        if random.random() < self.prob_elastic:
            image, mask = self._elastic_transform(image, mask)
            aug_count += 1

        if random.random() < self.prob_brightness:
            image = self._adjust_brightness(image)
            aug_count += 1

        if random.random() < self.prob_contrast:
            image = self._adjust_contrast(image)
            aug_count += 1

        if random.random() < self.prob_noise:
            image = self._add_gaussian_noise(image)
            aug_count += 1

        if random.random() < self.prob_blur:
            image = self._gaussian_blur(image)
            aug_count += 1

        if aug_count < self.min_augmentations:
            image, mask = self._force_augment(image, mask,
                                               self.min_augmentations - aug_count)

        if random.random() < self.double_aug_prob:
            if random.random() < 0.5:
                image = self._adjust_brightness(image)
            if random.random() < 0.5:
                image = self._adjust_contrast(image)
            if random.random() < 0.3:
                image = self._add_gaussian_noise(image)

        return image, mask

    def _force_augment(self, image, mask, num_augs):
        geometric_augs = ['rotation', 'scale', 'flip']
        intensity_augs = ['brightness', 'contrast', 'noise', 'blur']

        for _ in range(num_augs):
            if random.random() < 0.5:
                aug = random.choice(geometric_augs)
                if aug == 'rotation':
                    image, mask = self._random_rotation(image, mask)
                elif aug == 'scale':
                    image, mask = self._random_scale(image, mask)
                elif aug == 'flip':
                    image, mask = self._random_flip(image, mask)
            else:
                aug = random.choice(intensity_augs)
                if aug == 'brightness':
                    image = self._adjust_brightness(image)
                elif aug == 'contrast':
                    image = self._adjust_contrast(image)
                elif aug == 'noise':
                    image = self._add_gaussian_noise(image)
                elif aug == 'blur':
                    image = self._gaussian_blur(image)

        return image, mask

    def _random_rotation(self, image, mask):
        from scipy.ndimage import rotate
        angle = random.uniform(*self.rotation_range)
        
        if image.ndim == 2:
            image = rotate(image, angle, reshape=False, mode='reflect', order=1)
        else:
           
            rotated_channels = []
            for c in range(image.shape[2]):
                rotated_channels.append(rotate(image[:, :, c], angle, reshape=False, mode='reflect', order=1))
            image = np.stack(rotated_channels, axis=2)
        mask = rotate(mask, angle, reshape=False, mode='reflect', order=0)
        return image, mask

    def _random_scale(self, image, mask):
        import cv2
        scale = random.uniform(*self.scale_range)

        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[0], image.shape[1]

        new_h, new_w = int(h * scale), int(w * scale)

        
        if image.ndim == 2:
            image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        
        if image.ndim == 2:
            image_out = np.zeros((h, w), dtype=image.dtype)
        else:
            image_out = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
        mask_out = np.zeros((h, w), dtype=mask.dtype)

        if scale >= 1.0:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            if image.ndim == 2:
                image_out = image_scaled[start_h:start_h+h, start_w:start_w+w]
            else:
                image_out = image_scaled[start_h:start_h+h, start_w:start_w+w]
            mask_out = mask_scaled[start_h:start_h+h, start_w:start_w+w]
        else:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            if image.ndim == 2:
                image_out[start_h:start_h+new_h, start_w:start_w+new_w] = image_scaled
            else:
                image_out[start_h:start_h+new_h, start_w:start_w+new_w] = image_scaled
            mask_out[start_h:start_h+new_h, start_w:start_w+new_w] = mask_scaled

        return image_out, mask_out

    def _random_flip(self, image, mask):
        flip_code = random.choice([-1, 0, 1])  
        import cv2
        image = cv2.flip(image, flip_code)
        mask = cv2.flip(mask, flip_code)
        return image, mask

    def _elastic_transform(self, image, mask):
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[0], image.shape[1]

        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1) * self.elastic_alpha,
                             self.elastic_sigma)
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1) * self.elastic_alpha,
                             self.elastic_sigma)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (np.clip(y + dy, 0, h-1).astype(np.int32),
                   np.clip(x + dx, 0, w-1).astype(np.int32))

        if image.ndim == 2:
            image = image[indices]
        else:
            transformed_channels = []
            for c in range(image.shape[2]):
                transformed_channels.append(image[:, :, c][indices])
            image = np.stack(transformed_channels, axis=2)
        mask = mask[indices]

        return image, mask

    def _adjust_brightness(self, image):
        
        brightness = random.uniform(*self.brightness_range)
        image = np.clip(image * brightness, 0, 255)
        return image.astype(np.float32)

    def _adjust_contrast(self, image):
        
        contrast = random.uniform(*self.contrast_range)
        mean = image.mean()
        image = np.clip((image - mean) * contrast + mean, 0, 255)
        return image.astype(np.float32)

    def _add_gaussian_noise(self, image):
        
        noise = np.random.normal(0, self.gaussian_noise_std * 255, image.shape)
        image = np.clip(image + noise, 0, 255)
        return image.astype(np.float32)

    def _gaussian_blur(self, image):
        sigma = random.uniform(0.5, 1.5)
        if image.ndim == 2:
            return gaussian_filter(image, sigma=sigma).astype(np.float32)
        else:
            blurred_channels = []
            for c in range(image.shape[2]):
                blurred_channels.append(gaussian_filter(image[:, :, c], sigma=sigma).astype(np.float32))
            return np.stack(blurred_channels, axis=2)



def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def update_learning_rate(optimizer, epoch, args):
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lr = (epoch + 1) / args.warmup_epochs * args.lr
    elif args.scheduler == 'poly':
        effective_epoch = epoch - args.warmup_epochs if args.warmup_epochs > 0 else epoch
        effective_max_epochs = args.epochs - args.warmup_epochs if args.warmup_epochs > 0 else args.epochs
        lr = poly_lr(effective_epoch, effective_max_epochs, args.lr, args.poly_exponent)
    else:
        return optimizer.param_groups[0]['lr']

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



class ForegroundOversampleDataset(Dataset):
    

    def __init__(self, base_dataset, oversample_percent=0.33):
        self.base_dataset = base_dataset
        self.oversample_percent = oversample_percent
        self.foreground_indices = []

        print("Identifying foreground samples...")
        for idx in tqdm(range(len(base_dataset))):
            _, mask = base_dataset[idx]
            if mask.sum() > 0:
                self.foreground_indices.append(idx)

        print(f"Foreground samples: {len(self.foreground_indices)}/{len(base_dataset)}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if random.random() < self.oversample_percent and self.foreground_indices:
            fg_idx = random.choice(self.foreground_indices)
            return self.base_dataset[fg_idx]
        return self.base_dataset[idx]



class BCEDiceLoss(nn.Module):
    

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1 - dice

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class GradientFlowLoss(nn.Module):
   

    def __init__(self, mse_weight=0.5, cos_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight

    def forward(self, pred_flow, gt_flow, mask=None):
       
        mse_loss = F.mse_loss(pred_flow, gt_flow, reduction='none')

        if mask is not None:
            
            mask_binary = (mask > 0.5).float()
            mse_loss = (mse_loss * mask_binary).sum() / (mask_binary.sum() + 1e-8)
        else:
            mse_loss = mse_loss.mean()

       
        B, _, H, W = pred_flow.shape
        pred_flat = pred_flow.permute(0, 2, 3, 1).reshape(-1, 2)  
        gt_flat = gt_flow.permute(0, 2, 3, 1).reshape(-1, 2)

        
        cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1)
        cos_loss = (1 - cos_sim).mean()

        return self.mse_weight * mse_loss + self.cos_weight * cos_loss


class CellposeLoss(nn.Module):
    

    def __init__(self, seg_weight=0.5, flow_weight=0.5, use_flow_loss=True):
        super().__init__()
        self.seg_weight = seg_weight
        self.flow_weight = flow_weight
        self.use_flow_loss = use_flow_loss

        self.seg_loss = BCEDiceLoss()
        self.flow_loss = GradientFlowLoss()

    def forward(self, outputs, masks, flow_gt=None):
       
        seg_pred = outputs['seg']
        seg_loss = self.seg_loss(seg_pred, masks)

        
        if self.use_flow_loss and 'flow' in outputs and flow_gt is not None:
            flow_pred = outputs['flow']
            flow_loss = self.flow_loss(flow_pred, flow_gt, masks)
        else:
            flow_loss = torch.tensor(0.0, device=masks.device)

        
        total_loss = self.seg_weight * seg_loss + self.flow_weight * flow_loss

        return total_loss, seg_loss, flow_loss




def dice_coefficient(pred, target, threshold=0.5, is_sigmoid=False):
    
    if is_sigmoid:
        pred_binary = (pred > threshold).float()
    else:
        pred_binary = (torch.sigmoid(pred) > threshold).float()
    smooth = 1e-5
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_coefficient(pred, target, threshold=0.5, is_sigmoid=False):
   
    if is_sigmoid:
        pred_binary = (pred > threshold).float()
    else:
        pred_binary = (torch.sigmoid(pred) > threshold).float()
    smooth = 1e-5
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)




class TN3KDataset(Dataset):
   

    def __init__(self, image_dir, mask_dir, transform=None, img_size=256, is_train=True, augmentation=None, in_channels=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.is_train = is_train
        self.augmentation = augmentation
        self.in_channels = in_channels  

        self.image_files = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        print(f"Found {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        
        if self.in_channels == 1:
            image = Image.open(img_path).convert('L')  
        else:
            image = Image.open(img_path).convert('RGB')  
        image = np.array(image)

        base_name = os.path.splitext(img_name)[0]
        mask_name = None
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            candidate = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(candidate):
                mask_name = candidate
                break
            candidate = os.path.join(self.mask_dir, base_name + '_mask' + ext)
            if os.path.exists(candidate):
                mask_name = candidate
                break

        if mask_name is None:
            raise FileNotFoundError(f"Mask not found for {img_name}")

        mask = Image.open(mask_name).convert('L')
        mask = np.array(mask)

        mask = (mask > 127).astype(np.float32)

        
        if self.img_size > 0:
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)
        else:
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)

        
        if self.is_train and self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        
        if self.in_channels == 1:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-8)
            image = image.astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(0)  
        else:
            
            for c in range(3):
                mean = image[:, :, c].mean()
                std = image[:, :, c].std()
                image[:, :, c] = (image[:, :, c] - mean) / (std + 1e-8)
            image = image.astype(np.float32)
            image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class PatchDataset(Dataset):
   

    def __init__(self, image_dir, mask_dir, patch_size=256,
                 patches_per_image=10,
                 ensure_foreground_ratio=0.7,
                 in_channels=1, augmentation=None, is_train=True):
       
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.ensure_foreground_ratio = ensure_foreground_ratio
        self.in_channels = in_channels
        self.augmentation = augmentation
        self.is_train = is_train

        self.image_files = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

       
        self.images = {}
        self.masks = {}
        self.has_foreground = {}

        print(f"Preloading {len(self.image_files)} images...")
        for img_name in self.image_files:
            img_path = os.path.join(image_dir, img_name)
            if in_channels == 1:
                img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
            else:
                img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
            self.images[img_name] = img

            
            base_name = os.path.splitext(img_name)[0]
            mask_path = None
            for ext in ['.png', '.jpg', '.bmp', '.tif']:
                candidate = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {img_name}")

            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
            mask = (mask > 127).astype(np.float32)
            self.masks[img_name] = mask
            self.has_foreground[img_name] = mask.max() > 0

        print(f"PatchDataset: {len(self.image_files)} images, "
              f"{patches_per_image} patches per image, "
              f"{len(self.image_files) * patches_per_image} samples in total")

    def __len__(self):
        return len(self.image_files) * self.patches_per_image

    def __getitem__(self, idx):
        
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        img_name = self.image_files[img_idx]
        image = self.images[img_name].copy()
        mask = self.masks[img_name].copy()

        H, W = image.shape[:2]

        
        if self.is_train and random.random() < self.ensure_foreground_ratio and self.has_foreground[img_name]:
            
            foreground_coords = np.where(mask > 0)
            if len(foreground_coords[0]) > 0:
                
                rand_idx = random.randint(0, len(foreground_coords[0]) - 1)
                cy, cx = foreground_coords[0][rand_idx], foreground_coords[1][rand_idx]

                
                y_start = max(0, min(cy - self.patch_size // 2, H - self.patch_size))
                x_start = max(0, min(cx - self.patch_size // 2, W - self.patch_size))
            else:
                
                y_start = random.randint(0, max(0, H - self.patch_size))
                x_start = random.randint(0, max(0, W - self.patch_size))
        else:
            
            y_start = random.randint(0, max(0, H - self.patch_size))
            x_start = random.randint(0, max(0, W - self.patch_size))

        
        y_end = min(y_start + self.patch_size, H)
        x_end = min(x_start + self.patch_size, W)

        image_patch = image[y_start:y_end, x_start:x_end]
        mask_patch = mask[y_start:y_end, x_start:x_end]

        
        if image_patch.shape[0] < self.patch_size or image_patch.shape[1] < self.patch_size:
            if self.in_channels == 1:
                padded_image = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                padded_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            else:
                padded_image = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
                padded_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

            h, w = image_patch.shape[:2]
            padded_image[:h, :w] = image_patch if self.in_channels == 1 else image_patch
            padded_mask[:h, :w] = mask_patch

            image_patch = padded_image
            mask_patch = padded_mask

        
        if self.is_train and self.augmentation is not None:
            image_patch, mask_patch = self.augmentation(image_patch, mask_patch)

        
        if self.in_channels == 1:
            mean = image_patch.mean()
            std = image_patch.std()
            image_patch = (image_patch - mean) / (std + 1e-8)
            image_patch = torch.from_numpy(image_patch).unsqueeze(0)  
        else:
            for c in range(3):
                mean = image_patch[:, :, c].mean()
                std = image_patch[:, :, c].std()
                image_patch[:, :, c] = (image_patch[:, :, c] - mean) / (std + 1e-8)
            image_patch = torch.from_numpy(image_patch).permute(2, 0, 1)  

        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0)

        return image_patch, mask_patch


def sliding_window_inference(model, image, patch_size=256, stride=128,
                              blend_mode='average', device='cuda', use_flow=False,
                              verbose=False):
    
    model.eval()
    B, C, H, W = image.shape

    if verbose:
        print(f"  [Sliding window] Input size: {H}x{W}, patch: {patch_size}, stride: {stride}")

    
    output_seg = torch.zeros(B, 1, H, W, device=device)
    count = torch.zeros(B, 1, H, W, device=device)

    if use_flow:
        output_flow = torch.zeros(B, 2, H, W, device=device)
    else:
        output_flow = None

    
    if H <= patch_size and W <= patch_size:
        with torch.no_grad():
            outputs = model(image)
            pred = torch.sigmoid(outputs['seg'])
            if use_flow and 'flow' in outputs:
                return pred, outputs['flow']
            return pred, output_flow

    
    if H <= patch_size:
        y_positions = [0]
    else:
        y_positions = list(range(0, H - patch_size + 1, stride))
        if y_positions[-1] + patch_size < H:
            y_positions.append(H - patch_size)

    if W <= patch_size:
        x_positions = [0]
    else:
        x_positions = list(range(0, W - patch_size + 1, stride))
        if x_positions[-1] + patch_size < W:
            x_positions.append(W - patch_size)

    if verbose:
        print(f"  [Sliding window] y positions: {len(y_positions)}, x positions: {len(x_positions)}, total patches: {len(y_positions)*len(x_positions)}")

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                outputs = model(patch)

                pred_seg = torch.sigmoid(outputs['seg'])

                output_seg[:, :, y:y+patch_size, x:x+patch_size] += pred_seg
                count[:, :, y:y+patch_size, x:x+patch_size] += 1

                if use_flow and 'flow' in outputs:
                    output_flow[:, :, y:y+patch_size, x:x+patch_size] += outputs['flow']

    
    count = count.clamp(min=1)
    output_seg = output_seg / count

    if use_flow and output_flow is not None:
        output_flow = output_flow / count

    return output_seg, output_flow




def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs,
                    max_grad_norm=1.0, use_flow_loss=True):
    
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_flow_loss = 0
    total_dice = 0
    valid_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                leave=False, ncols=100)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        
        if use_flow_loss:
            flow_gt = compute_flow_gt_batch(masks, device)
        else:
            flow_gt = None

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)

            
            if 'seg_ds' in outputs:
                loss, seg_loss, flow_loss_val = criterion(outputs, masks, flow_gt)

                
                outputs_ds = {'seg': outputs['seg_ds']}
                if 'flow_ds' in outputs:
                    outputs_ds['flow'] = outputs['flow_ds']
                if flow_gt is not None:
                    flow_gt_ds = F.interpolate(flow_gt,
                                                size=outputs['seg_ds'].shape[2:],
                                                mode='bilinear', align_corners=True)
                else:
                    flow_gt_ds = None
                masks_ds = F.interpolate(masks,
                                          size=outputs['seg_ds'].shape[2:],
                                          mode='nearest')

                loss_ds, _, _ = criterion(outputs_ds, masks_ds, flow_gt_ds)
                loss = loss + 0.5 * loss_ds
            else:
                loss, seg_loss, flow_loss_val = criterion(outputs, masks, flow_gt)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: Batch {batch_idx} has NaN/Inf loss, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_flow_loss += flow_loss_val.item() if isinstance(flow_loss_val, torch.Tensor) else flow_loss_val
        total_dice += dice_coefficient(outputs['seg'], masks).item()
        valid_batches += 1

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Seg': f'{seg_loss.item():.4f}',
            'Flow': f'{flow_loss_val.item() if isinstance(flow_loss_val, torch.Tensor) else 0:.4f}',
            'Dice': f'{dice_coefficient(outputs["seg"], masks).item():.4f}'
        })

    if valid_batches == 0:
        return float('nan'), 0.0, 0.0, 0.0

    return (total_loss / valid_batches, total_seg_loss / valid_batches,
            total_flow_loss / valid_batches, total_dice / valid_batches)


def validate(model, dataloader, criterion, device, epoch, total_epochs, use_flow_loss=True,
             use_sliding_window=False, patch_size=256, stride=128):
   
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_flow_loss = 0
    total_dice = 0
    total_iou = 0
    valid_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]  ",
                leave=False, ncols=100)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            if use_sliding_window:
               
                verbose = (batch_idx == 0 and epoch == 0)
                pred_seg, pred_flow = sliding_window_inference(
                    model, images, patch_size=patch_size, stride=stride,
                    device=device, use_flow=use_flow_loss, verbose=verbose
                )
                
                pred_seg_logits = torch.log(pred_seg / (1 - pred_seg + 1e-8) + 1e-8)
                outputs = {'seg': pred_seg_logits}
                if pred_flow is not None:
                    outputs['flow'] = pred_flow
                
                preds = pred_seg
                is_sigmoid = True
            else:
                
                outputs = model(images)
                preds = torch.sigmoid(outputs['seg'])
                is_sigmoid = True  

            if use_flow_loss:
                flow_gt = compute_flow_gt_batch(masks, device)
            else:
                flow_gt = None

            loss, seg_loss, flow_loss_val = criterion(outputs, masks, flow_gt)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_flow_loss += flow_loss_val.item() if isinstance(flow_loss_val, torch.Tensor) else flow_loss_val
            total_dice += dice_coefficient(preds, masks, is_sigmoid=is_sigmoid).item()
            total_iou += iou_coefficient(preds, masks, is_sigmoid=is_sigmoid).item()
            valid_batches += 1

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice_coefficient(preds, masks, is_sigmoid=is_sigmoid).item():.4f}'
            })

    if valid_batches == 0:
        return float('nan'), 0.0, 0.0, 0.0, 0.0

    return (total_loss / valid_batches, total_seg_loss / valid_batches,
            total_flow_loss / valid_batches, total_dice / valid_batches, total_iou / valid_batches)



def plot_training_curves(history: dict, save_path: str):
   
    epochs = history['epoch']
    if len(epochs) < 1:
        print("No training history, skipping curve plotting")
        return

    base_dir = os.path.dirname(save_path)
    os.makedirs(base_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(save_path))[0]


    def _plot_loss(ax):
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if history['val_loss']:
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train / Val Loss')
        ax.legend(loc='best', framealpha=1.0, handlelength=2, labelspacing=0.5)
        ax.grid(True, alpha=0.3)


    def _plot_convergence(ax):
        val_dice_list = history['val_dice'] if history['val_dice'] else []
        if not val_dice_list:
            ax.set_title('Val Dice Convergence (no data)')
            return

        ax.plot(epochs, val_dice_list, 'b-', label='Val Dice', linewidth=1.5, alpha=0.6)

        best_so_far = []
        best_val = float('-inf')
        for d in val_dice_list:
            best_val = max(best_val, d)
            best_so_far.append(best_val)
        ax.plot(epochs, best_so_far, 'g-', label='Best Val Dice (convergence)', linewidth=2.5)

        final_best = max(best_so_far)
        final_idx = best_so_far.index(final_best)
        ax.annotate(f'Best: {final_best:.4f}\nEpoch {epochs[final_idx]+1}',
                     xy=(epochs[final_idx], final_best),
                     xytext=(epochs[final_idx] + max(1, len(epochs) // 15), final_best),
                     fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Dice')
        ax.set_title('Validation Segmentation Convergence')
        ax.legend(loc='best', framealpha=1.0, handlelength=2, labelspacing=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)


    def _plot_lr(ax):
        lr_list = history.get('lr', [])
        if lr_list:
            ax.plot(epochs, lr_list, 'k-', label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend(loc='best', framealpha=1.0, handlelength=2, labelspacing=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')


    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    _plot_loss(axes[0])
    _plot_convergence(axes[1])
    _plot_lr(axes[2])
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

 
    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_loss(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, f'{base_name}_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_convergence(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, f'{base_name}_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7.5, 5))
    _plot_lr(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(base_dir, f'{base_name}_lr.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Training curves saved to:")
    print(f"  - Overview:      {save_path}")
    print(f"  - Loss curve:    {os.path.join(base_dir, f'{base_name}_loss.png')}")
    print(f"  - Convergence:   {os.path.join(base_dir, f'{base_name}_convergence.png')}")
    print(f"  - LR schedule:   {os.path.join(base_dir, f'{base_name}_lr.png')}")




def parse_args():
    parser = argparse.ArgumentParser(description='SGWDSNet Training')

   
    parser.add_argument('--image_dir', type=str, default=r'E:\picture\trainval-image',
                        help='training image directory')
    parser.add_argument('--mask_dir', type=str, default=r'E:\picture\trainval-mask',
                        help='training mask directory')
    parser.add_argument('--val_image_dir', type=str, default=None,
                        help='validation image directory')
    parser.add_argument('--val_mask_dir', type=str, default=None,
                        help='validation mask directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_cellpose',
                        help='checkpoint output directory')


    parser.add_argument('--in_channels', type=int, default=1, help='input channels')
    parser.add_argument('--n_channels', type=int, default=32, help='base channels')
    parser.add_argument('--n_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--img_size', type=int, default=256, help='input image size')


    parser.add_argument('--predict_flow', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True, help='predict gradient flow')
    parser.add_argument('--predict_cell_prob', type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=False, help='predict cell probability')
    parser.add_argument('--flow_weight', type=float, default=0.5,
                        help='flow loss weight')
    parser.add_argument('--seg_weight', type=float, default=0.5,
                        help='segmentation loss weight')


    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')


    parser.add_argument('--scheduler', type=str, default='poly',
                        choices=['cosine', 'step', 'plateau', 'poly'])
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--poly_exponent', type=float, default=0.9)

  
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--deep_supervision', action='store_true', default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--use_l2_pool', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='use L2 pooling in MSAS (True/False)')
    parser.add_argument('--oversample_foreground', action='store_true', default=True)
    parser.add_argument('--oversample_foreground_percent', type=float, default=0.33)

   
    parser.add_argument('--use_patch_training', action='store_true', default=False,
                        help='use patch training (random crop)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='patch size for patch training and sliding window inference')
    parser.add_argument('--patches_per_image', type=int, default=10,
                        help='number of patches cropped per image')
    parser.add_argument('--ensure_foreground_ratio', type=float, default=0.7,
                        help='probability of cropping a foreground patch (0-1)')
    parser.add_argument('--stride', type=int, default=128,
                        help='sliding window stride for validation inference')

    return parser.parse_args()




def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    train_augmentation = OnlineAugmentation(
        rotation_range=(-30, 30),
        scale_range=(0.7, 1.3),
        brightness_range=(0.7, 1.3),
        contrast_range=(0.7, 1.3),
        gaussian_noise_std=0.05,
        elastic_alpha=100,
        elastic_sigma=10,
        prob_rotation=0.7,
        prob_scale=0.6,
        prob_flip=0.5,
        prob_brightness=0.4,
        prob_contrast=0.4,
        prob_noise=0.4,
        prob_blur=0.3,
        prob_elastic=0.4,
        min_augmentations=2,
        double_aug_prob=0.3
    )

    print("Loading dataset...")


    if args.use_patch_training:
        print(f"Using Patch training mode: patch_size={args.patch_size}, "
              f"patches_per_image={args.patches_per_image}, "
              f"ensure_foreground_ratio={args.ensure_foreground_ratio}")
        train_dataset = PatchDataset(
            args.image_dir, args.mask_dir,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            ensure_foreground_ratio=args.ensure_foreground_ratio,
            in_channels=args.in_channels,
            augmentation=train_augmentation,
            is_train=True
        )
    else:
        train_dataset = TN3KDataset(args.image_dir, args.mask_dir, img_size=args.img_size,
                                    is_train=True, augmentation=train_augmentation,
                                    in_channels=args.in_channels)

    if args.oversample_foreground and not args.use_patch_training:
        train_dataset = ForegroundOversampleDataset(train_dataset, args.oversample_foreground_percent)

    if args.val_image_dir and args.val_mask_dir:
        print("Using independent validation set...")
        if args.use_patch_training:
           
            val_dataset = TN3KDataset(args.val_image_dir, args.val_mask_dir, img_size=0,
                                      is_train=False, augmentation=None,
                                      in_channels=args.in_channels)
        else:
            val_dataset = TN3KDataset(args.val_image_dir, args.val_mask_dir, img_size=args.img_size,
                                      is_train=False, augmentation=None,
                                      in_channels=args.in_channels)
        print(f"Train set: {len(train_dataset)} samples, validation set: {len(val_dataset)} images")
    else:
        val_size = int(len(train_dataset) * args.val_ratio)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                                   generator=torch.Generator().manual_seed(args.seed))
        print(f"Train set: {len(train_dataset)} images, validation set: {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)


    val_batch_size = 1 if args.use_patch_training else args.batch_size
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Creating model...")
    model = SGWDSNet(
        in_channels=args.in_channels,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        predict_flow=args.predict_flow,
        predict_cell_prob=args.predict_cell_prob,
        deep_supervision=args.deep_supervision,
        dropout_rate=args.dropout_rate,
        use_l2_pool=args.use_l2_pool
    ).to(device)

    criterion = CellposeLoss(
        seg_weight=args.seg_weight,
        flow_weight=args.flow_weight,
        use_flow_loss=args.predict_flow
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                          eta_min=args.min_lr)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          patience=10, factor=0.5)
    else:
        scheduler = None

    scaler = GradScaler()

    start_epoch = 0
    best_dice = 0


    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_seg_loss': [],
        'train_flow_loss': [],
        'val_seg_loss': [],
        'val_flow_loss': [],
        'val_iou': [],
        'lr': []
    }

    if args.resume:
        print(f"Resuming training from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0)
        
        if 'history' in checkpoint and checkpoint['history']:
            history = checkpoint['history']
            print(f"Restored training history: {len(history['epoch'])} epochs")

    print("SGWDSNet training configuration:")
    print("=" * 60)
    print(f"  Input channels: {args.in_channels} ({'grayscale' if args.in_channels == 1 else 'RGB'})")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Predict gradient flow: {args.predict_flow}")
    print(f"  Flow loss weight: {args.flow_weight}")
    print(f"  Segmentation loss weight: {args.seg_weight}")
    print(f"  Deep supervision: {args.deep_supervision}")
    print(f"  Modules: MSAS + SC2A + SGWDS + GASG")
    print(f"  L2 pooling: {args.use_l2_pool}")
    print(f"  Patch training mode: {args.use_patch_training}")
    if args.use_patch_training:
        print(f"  Patch size: {args.patch_size}")
        print(f"  Patches per image: {args.patches_per_image}")
        print(f"  Ensure foreground ratio: {args.ensure_foreground_ratio}")
        print(f"  Sliding window stride: {args.stride}")
    print("=" * 60 + "\n")

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_seg_loss, train_flow_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs,
            use_flow_loss=args.predict_flow
        )

        val_loss, val_seg_loss, val_flow_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device, epoch, args.epochs,
            use_flow_loss=args.predict_flow,
            use_sliding_window=args.use_patch_training,  
            patch_size=args.patch_size,
            stride=args.stride
        )

        if args.scheduler == 'poly' or args.warmup_epochs > 0:
            current_lr = update_learning_rate(optimizer, epoch, args)
        elif scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Flow: {train_flow_loss:.4f}), Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f} | "
              f"LR: {current_lr:.6f}")


        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_seg_loss'].append(train_seg_loss)
        history['train_flow_loss'].append(train_flow_loss)
        history['val_seg_loss'].append(val_seg_loss)
        history['val_flow_loss'].append(val_flow_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)


        if (epoch + 1) == args.epochs:
            curves_path = os.path.join(args.output_dir, 'training_curves.png')
            plot_training_curves(history, curves_path)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/seg_train', train_seg_loss, epoch)
        writer.add_scalar('Loss/flow_train', train_flow_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_iou': val_iou,
                'history': history,
            }, save_path)
            print(f"Saved best model: Dice={best_dice:.4f}")

        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_path)

    writer.close()
    print(f"\nTraining finished. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
