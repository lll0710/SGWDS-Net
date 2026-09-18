# 🦈 LGFFM: A Localized and Globalized Frequency Fusion Model for Ultrasound Image Segmentation

## 🐟 Dataset Format 🐟
```text
data/
├── train/
│   ├── images/
│   │   ├── case_0001.nii.gz
│   │   ├── case_0002.nii.gz
│   │   └── ...
│   └── masks/
│       ├── case_0001.nii.gz
│       ├── case_0002.nii.gz
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**🐟🐟🐟 Notes 🐟🐟🐟:**

1. **Validation Set**: It is recommended to provide an independent validation set (`val/`). If the `val/` directory is missing or empty, the data loader will automatically split a portion of the training set as the validation set.
2. **Filename Matching**: The filenames of the images and their corresponding masks must be identical (e.g., `case_0001.nii.gz`).

##  🐋Training
🐋 **1. Grayscale Images (Long or Width < 1000)**

**Datasets: BUSI, TN3K, DDTI, etc.**

```bash
python train.py \
--image_dir ./BUSI/train/images \
--mask_dir ./BUSI/train/masks \
--val_image_dir ./BUSI/val/images \
--val_mask_dir ./BUSI/val/masks \
--gpu 0 \
--output_dir ./checkpoints/BUSI/best_model.pth  \
```

🐋 **2. RGB Images (Long or Width < 1000)**

**CVC-ColonDB, CVC-ClinicDB, etc.**

```bash
python train.py \
--image_dir ./CVC-ColonDB/train/images \
--mask_dir ./CVC-ColonDB/train/masks \
--val_image_dir ./CVC-ColonDB/val/images \
--val_mask_dir ./CVC-ColonDB/val/masks \
--gpu 0 \
--in_channels 3 \
--output_dir ./checkpoints/CVC-ColonDB/best_model.pth  \
```

🐋 **3. RGB Images (Long or Width >= 1000)**

**Datasets: MoNuSeg, CoNSeP, etc.**

```bash
python train.py \
--image_dir ./MoNuSeg/train/images \
--mask_dir ./MoNuSeg/train/masks \
--val_image_dir ./MoNuSeg/val/images \
--val_mask_dir ./MoNuSeg/val/masks \
--gpu 0 \
--use_patch_training \
--patch_size 256 \
--patches_per_image 10 \
--ensure_foreground_ratio 0.7 \
--stride 128 \
--in_channels 3 \
--output_dir ./checkpoints/CVC-ColonDB/best_model.pth  \
```

## 🐡 Pretrained Weights

If you want to directly test with our trained weights, please download them from Baidu Netdisk:

- **Baidu Netdisk:** [Download](https://pan.baidu.com/s/1o90ZiBiucLmNtv_ZE_76aA) (Access Code: yu71)

After downloading, please put the weights in the `checkpoints/` directory.

