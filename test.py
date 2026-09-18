# -----------------------------------------------------------
# test.py - SGWDSNet 测试脚本
# 功能：
# 1. 加载训练好的模型
# 2. 批量测试 2D 医学图像数据集
# 3. 评估指标 (分割 + 梯度流场)
# 4. 文本结果导出
# -----------------------------------------------------------
import os
import argparse
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SGWDSNet import SGWDSNet, compute_gradient_flow_from_mask


# ===================== 滑动窗口推理 =====================

def sliding_window_inference(model, image, patch_size=256, stride=128,
                              blend_mode='average', device='cuda', use_flow=False):
    """
    滑动窗口推理

    Args:
        model: 模型
        image: [B, C, H, W] 输入图像
        patch_size: patch大小
        stride: 滑动步长
        blend_mode: 重叠区域融合方式 ('average' 或 'max')
        device: 计算设备
        use_flow: 是否输出流场

    Returns:
        pred_mask: [B, 1, H, W] 预测mask
        pred_flow: [B, 2, H, W] 预测流场 (如果use_flow=True)
    """
    model.eval()
    B, C, H, W = image.shape

    # 初始化输出和计数图
    output_seg = torch.zeros(B, 1, H, W, device=device)
    count = torch.zeros(B, 1, H, W, device=device)

    if use_flow:
        output_flow = torch.zeros(B, 2, H, W, device=device)
    else:
        output_flow = None

    # 处理图像小于patch_size的情况
    if H < patch_size or W < patch_size:
        with torch.no_grad():
            outputs = model(image)
            pred = torch.sigmoid(outputs['seg'])
            if use_flow and 'flow' in outputs:
                return pred, outputs['flow']
            return pred, output_flow

    # 滑动窗口
    y_positions = list(range(0, H - patch_size + 1, stride))
    x_positions = list(range(0, W - patch_size + 1, stride))

    # 确保覆盖到边缘
    if y_positions[-1] + patch_size < H:
        y_positions.append(H - patch_size)
    if x_positions[-1] + patch_size < W:
        x_positions.append(W - patch_size)

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

    # 平均融合
    count = count.clamp(min=1)
    output_seg = output_seg / count

    if use_flow and output_flow is not None:
        output_flow = output_flow / count

    return output_seg, output_flow


# ===================== 配置参数 =====================

def parse_args():
    parser = argparse.ArgumentParser(description='SGWDSNet Testing')

    parser.add_argument('--image_dir', type=str, default=r'E:\picture\trainval-image',
                        help='测试图像目录')
    parser.add_argument('--mask_dir', type=str, default=r'E:\picture\trainval-mask',
                        help='测试标签目录')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果保存目录')

    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--in_channels', type=int, default=1, help='输入通道数')
    parser.add_argument('--n_channels', type=int, default=32, help='基础通道数')
    parser.add_argument('--n_classes', type=int, default=1, help='输出类别数')
    parser.add_argument('--predict_flow', action='store_true', default=True,
                        help='是否预测梯度流场')
    parser.add_argument('--predict_cell_prob', action='store_true', default=False,
                        help='是否预测细胞概率')
    parser.add_argument('--deep_supervision', action='store_true', default=False,
                        help='是否使用深层监督')

    parser.add_argument('--img_size', type=int, default=256, help='输入图像大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='分割阈值')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID，-1表示CPU')
    parser.add_argument('--use_l2_pool', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
                        help='是否在 MSAS 中使用 L2 池化（True/False），需与训练时一致')

    parser.add_argument('--max_samples', type=int, default=0,
                        help='最多测试几张图（0表示全部）')
    parser.add_argument('--image_names', type=str, default='',
                        help='指定测试的图片名称，多个用逗号分隔')

    # 滑动窗口推理参数
    parser.add_argument('--use_sliding_window', action='store_true', default=False,
                        help='是否使用滑动窗口推理（用于大图像）')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='滑动窗口的patch大小')
    parser.add_argument('--stride', type=int, default=128,
                        help='滑动窗口的步长')

    return parser.parse_args()


# ===================== 数据集 =====================

class MedicalDataset(Dataset):
    """医学图像分割数据集"""

    def __init__(self, image_dir, mask_dir, img_size=256, image_names=None, max_samples=0, in_channels=1, use_sliding_window=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.in_channels = in_channels
        self.use_sliding_window = use_sliding_window  # 新增：是否使用滑动窗口推理

        self.all_images = [f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

        if image_names:
            names_set = set(image_names) if isinstance(image_names, (list, set)) else set(
                n.strip() for n in image_names.split(',') if n.strip())
            self.all_images = [f for f in self.all_images if f in names_set]

        if max_samples > 0:
            self.all_images = self.all_images[:max_samples]

        self.image_files = self.all_images
        print(f"找到 {len(self.image_files)} 张图像 (输入通道: {in_channels})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 根据 in_channels 决定读取方式
        if self.in_channels == 1:
            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            original_h, original_w = original_img.shape
            image = Image.open(img_path).convert('L')
        else:
            original_img = cv2.imread(img_path)
            original_h, original_w = original_img.shape[:2]
            image = Image.open(img_path).convert('RGB')

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

        if mask_name:
            mask = Image.open(mask_name).convert('L')
        else:
            mask = Image.new('L', (original_w, original_h), 0)

        # 滑动窗口模式：保持原始尺寸，不resize
        if not self.use_sliding_window:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image_np = np.array(image, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.float32)

        mask_np = (mask_np > 127).astype(np.float32)

        # 归一化 - 对每个通道分别处理
        if self.in_channels == 1:
            mean = image_np.mean()
            std = image_np.std()
            image_np = (image_np - mean) / (std + 1e-8)
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W]
        else:
            # RGB: 对每个通道分别归一化
            for c in range(3):
                mean = image_np[:, :, c].mean()
                std = image_np[:, :, c].std()
                image_np[:, :, c] = (image_np[:, :, c] - mean) / (std + 1e-8)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [3, H, W]

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {
            'image': image_tensor,
            'label': mask_tensor,
            'case': base_name,
            'image_path': img_path,
            'mask_path': mask_name if mask_name else '',
            'original_size': (original_h, original_w)
        }


# ===================== 评估指标 =====================

def dice_coefficient(pred, target, threshold=0.5):
    """计算Dice系数"""
    pred = (pred > threshold).float()
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_coefficient(pred, target, threshold=0.5):
    """计算IoU"""
    pred = (pred > threshold).float()
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred, target, threshold=0.5):
    """计算Precision = TP / (TP + FP)"""
    pred = (pred > threshold).float()
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred, target, threshold=0.5):
    """计算Recall = TP / (TP + FN)"""
    pred = (pred > threshold).float()
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred, target, threshold=0.5):
    """计算Specificity = TN / (TN + FP)"""
    pred = (pred > threshold).float()
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def flow_mse(pred_flow, gt_flow):
    """计算梯度流场 MSE"""
    return F.mse_loss(pred_flow, gt_flow)


def flow_cosine_similarity(pred_flow, gt_flow):
    """计算梯度流场余弦相似度"""
    B = pred_flow.shape[0]
    cos_sim = F.cosine_similarity(pred_flow, gt_flow, dim=1)
    return cos_sim.mean()


# ===================== 批量测试 =====================

def test_batch(model, dataloader, device, threshold=0.5, args=None):
    """批量测试并计算指标"""
    model.eval()

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_specificity = []
    all_flow_mse = []
    all_flow_cosine = []
    all_loss = []
    sample_results = []  # 新增：存储每张图的指标用于CSV导出

    def bce_dice_loss(pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        return bce + (1 - dice)

    pbar = tqdm(dataloader, desc="Testing", ncols=100)

    use_sliding_window = args is not None and getattr(args, 'use_sliding_window', False)
    if use_sliding_window:
        print(f"使用滑动窗口推理: patch_size={args.patch_size}, stride={args.stride}")

    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['label'].to(device)
            case_name = batch['case'][0]
            mask_path = batch['mask_path'][0]

            if use_sliding_window:
                pred_seg, pred_flow = sliding_window_inference(
                    model, images,
                    patch_size=args.patch_size,
                    stride=args.stride,
                    device=device,
                    use_flow=True
                )
                seg_logits = pred_seg
                preds = pred_seg
                outputs = None
            else:
                outputs = model(images)
                seg_logits = outputs['seg'] if isinstance(outputs, dict) else outputs
                pred_flow = outputs.get('flow', None)
                preds = torch.sigmoid(seg_logits)

            if use_sliding_window:
                loss = F.binary_cross_entropy(preds, masks).item()
            else:
                loss = bce_dice_loss(seg_logits, masks).item()

            dice = dice_coefficient(preds, masks, threshold).item()
            iou = iou_coefficient(preds, masks, threshold).item()
            prec = precision_score(preds, masks, threshold).item()
            rec = recall_score(preds, masks, threshold).item()
            spec = specificity_score(preds, masks, threshold).item()

            all_dice.append(dice)
            all_iou.append(iou)
            all_precision.append(prec)
            all_recall.append(rec)
            all_specificity.append(spec)
            all_loss.append(loss)

            flow_mse_val = 0.0
            flow_cos_val = 0.0

            if pred_flow is not None and mask_path and os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask_binary = (gt_mask > 127).astype(np.float32)
                    H, W = pred_flow.shape[2], pred_flow.shape[3]
                    gt_mask_resized = cv2.resize(gt_mask_binary, (W, H), interpolation=cv2.INTER_NEAREST)
                    flow_gt_np = compute_gradient_flow_from_mask(gt_mask_resized)
                    flow_gt = torch.from_numpy(flow_gt_np).unsqueeze(0).to(device)
                    flow_mse_val = flow_mse(pred_flow, flow_gt).item()
                    flow_cos_val = flow_cosine_similarity(pred_flow, flow_gt).item()
                    all_flow_mse.append(flow_mse_val)
                    all_flow_cosine.append(flow_cos_val)

            # 记录单样本结果
            sample_results.append({
                'case_name': case_name,
                'dice': round(dice, 4),
                'iou': round(iou, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'specificity': round(spec, 4),
                'flow_mse': round(flow_mse_val, 6),
                'flow_cosine': round(flow_cos_val, 4),
                'loss': round(loss, 4)
            })

            pbar.set_postfix({
                'Dice': f'{np.mean(all_dice):.4f}',
                'IoU': f'{np.mean(all_iou):.4f}'
            })

            del images, masks, preds
            if outputs is not None:
                del outputs
            if pred_flow is not None:
                del pred_flow
            torch.cuda.empty_cache()

    results = {
        'Dice': {'mean': np.mean(all_dice), 'std': np.std(all_dice),
                 'min': np.min(all_dice), 'max': np.max(all_dice)},
        'IoU': {'mean': np.mean(all_iou), 'std': np.std(all_iou),
                'min': np.min(all_iou), 'max': np.max(all_iou)},
        'Precision': {'mean': np.mean(all_precision), 'std': np.std(all_precision),
                     'min': np.min(all_precision), 'max': np.max(all_precision)},
        'Recall': {'mean': np.mean(all_recall), 'std': np.std(all_recall),
                   'min': np.min(all_recall), 'max': np.max(all_recall)},
        'Specificity': {'mean': np.mean(all_specificity), 'std': np.std(all_specificity),
                       'min': np.min(all_specificity), 'max': np.max(all_specificity)},
        'Loss': {'mean': np.mean(all_loss), 'std': np.std(all_loss)},
        'Flow_MSE': {'mean': np.mean(all_flow_mse) if all_flow_mse else 0,
                     'std': np.std(all_flow_mse) if all_flow_mse else 0},
        'Flow_Cosine': {'mean': np.mean(all_flow_cosine) if all_flow_cosine else 0,
                        'std': np.std(all_flow_cosine) if all_flow_cosine else 0},
        'sample_results': sample_results  # 新增：返回单样本结果
    }

    return results


def save_results_to_csv(sample_results, save_path):
    """将单样本结果保存为 CSV 文件"""
    if not sample_results:
        print("无结果可保存")
        return

    keys = ['case_name', 'dice', 'iou', 'precision', 'recall',
            'specificity', 'flow_mse', 'flow_cosine', 'loss']

    with open(save_path, 'w', newline='') as f:
        f.write(','.join(keys) + '\n')
        for r in sample_results:
            f.write(','.join(str(r[k]) for k in keys) + '\n')

    print(f"CSV 结果已保存到: {save_path}")


# ===================== 主函数 =====================

def main():
    args = parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用 GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")

    print("创建模型...")
    model = SGWDSNet(
        in_channels=args.in_channels,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        predict_flow=args.predict_flow,
        predict_cell_prob=args.predict_cell_prob,
        deep_supervision=args.deep_supervision,
        use_l2_pool=args.use_l2_pool
    ).to(device)

    if os.path.exists(args.model_path):
        print(f"加载模型权重: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"模型来自 Epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"警告: 模型文件 {args.model_path} 不存在，使用随机初始化权重")

    dataset = MedicalDataset(args.image_dir, args.mask_dir, img_size=args.img_size,
                             image_names=args.image_names, max_samples=args.max_samples,
                             in_channels=args.in_channels,
                             use_sliding_window=args.use_sliding_window)  # 传入滑动窗口参数
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print("\n批量测试...")
    results = test_batch(model, dataloader, device, args.threshold, args=args)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)

    for metric_name in ['Dice', 'IoU', 'Precision', 'Recall', 'Specificity']:
        values = results[metric_name]
        print(f"\n{metric_name}:")
        print(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}")
        print(f"  Min:  {values['min']:.4f}")
        print(f"  Max:  {values['max']:.4f}")

    print(f"\nFlow MSE: {results['Flow_MSE']['mean']:.6f}")
    print(f"Flow Cosine Similarity: {results['Flow_Cosine']['mean']:.4f}")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存 CSV 结果
    if 'sample_results' in results and results['sample_results']:
        csv_path = os.path.join(args.output_dir, 'test_results.csv')
        save_results_to_csv(results['sample_results'], csv_path)

    result_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write("SGWDSNet 测试结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"图像大小: {args.img_size}\n")
        f.write(f"阈值: {args.threshold}\n")
        f.write("=" * 60 + "\n\n")
        for metric_name in ['Dice', 'IoU', 'Precision', 'Recall', 'Specificity']:
            values = results[metric_name]
            f.write(f"{metric_name}:\n")
            f.write(f"  Mean: {values['mean']:.4f} ± {values['std']:.4f}\n")
            f.write(f"  Min:  {values['min']:.4f}\n")
            f.write(f"  Max:  {values['max']:.4f}\n")
            f.write("\n")
        f.write(f"Flow MSE: {results['Flow_MSE']['mean']:.6f}\n")
        f.write(f"Flow Cosine: {results['Flow_Cosine']['mean']:.4f}\n")

    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
