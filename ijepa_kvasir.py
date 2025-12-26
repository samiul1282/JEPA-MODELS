# %%
# ==============================================================
# 0.  Setup (run once per machine)
# ==============================================================
# !pip install -q torch torchvision albumentations timm torchinfo pyyaml tqdm pillow
# !wget -q https://github.com/BRISC-Dataset/BRISC2025/archive/refs/heads/main.zip
# !unzip -q main.zip && mv BRISC2025-main brisc2025

# %%
# ==============================================================
# 1.  Imports
# ==============================================================
import os, yaml, cv2, math, random, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter 
import timm
from torchinfo import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# # 2.  Hyper-parameters & paths

# %%
cfg = {
    'img_size': 448,
    'patch_size': 16,
    'num_classes': 2,                 # background + tumor
    'batch_size': 16,                 # reduce if OOM
    'lr': 1e-3,
    'epochs': 30,
    'weight_decay': 1e-4,
    'num_workers': 0,
    'seed': 42,
    'root': Path('./data'),
    'weights': 'IN1K-vit.h.16-448px-300e.pth.tar',  # I-JEPA ckpt
}
# Download frozen I-JEPA weights once:
# !wget -q https://github.com/facebookresearch/ijepa/releases/download/v1.0/IN1K-vit.h.16-448px-300e.pth.tar

# %% [markdown]
# # 3.  Reproducibility

# %%
# ==============================================================

# ==============================================================
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(cfg['seed'])

# %% [markdown]
# # 4.  Dataset

# %%
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms

        # collect all image files (jpg / jpeg / png) once
        self.files = sorted(
            list(self.img_dir.glob('*.jpg'))  +
            list(self.img_dir.glob('*.jpeg')) +
            list(self.img_dir.glob('*.png'))
        )
        if not self.files:
            raise RuntimeError(f'No JPG/PNG images found in {self.img_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        name     = img_path.stem
        ext      = img_path.suffix  # .jpg  .png  etc

        # read image
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f'Cannot read image {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read mask with same stem *and* same extension
        mask_path = self.mask_dir / f'{name}{ext}'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:          # fallback: try other extensions
            for try_ext in ('.png', '.jpg', '.jpeg'):
                mask_path = self.mask_dir / f'{name}{try_ext}'
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    break
        if mask is None:
            raise RuntimeError(f'No mask found for image {img_path}')

        mask = (mask > 200).astype(np.uint8)

        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        return img, mask.long()

train_aug = A.Compose([
    A.Resize(cfg['img_size'], cfg['img_size']),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])
val_aug = A.Compose([
    A.Resize(cfg['img_size'], cfg['img_size']),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2()
])

train_ds = SegDataset(cfg['root']/"Kvasir-SEG"/'images', cfg['root']/"Kvasir-SEG"/'masks', train_aug)
val_ds   = SegDataset(cfg['root']/"ETIS-Larib"/'images',  cfg['root']/"ETIS-Larib"/'masks',  val_aug)

train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                          num_workers=cfg['num_workers'], pin_memory=True)

# %% [markdown]
# # 5.  Model

# %%
# ==============================================================
# 5.  Model  (fixed for your timm list)
# ==============================================================
# 5-a  Load the I-JEPA backbone that *is* in your timm registry
backbone = timm.create_model(
    'vit_huge_patch16_gap_448.in1k_ijepa',  # <-- correct tag in your list
    pretrained=True,                        # loads the I-JEPA weights
    num_classes=0                           # drop cls head
)

# 5-b  (optional) overwrite with your local checkpoint if you want
# ckpt = torch.load(cfg['weights'], map_location='cpu')['encoder']
# ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
# backbone.load_state_dict(ckpt, strict=True)

# 5-c  Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False
backbone.eval().to(device)

# 5-d  Light-weight decode head (unchanged)
class Decoder(nn.Module):
    def __init__(self, in_ch, num_cls):
        super().__init__()
        self.head = nn.Conv2d(in_ch, num_cls, kernel_size=3, padding=1)
    def forward(self, x):
        return self.head(x)

class JepaSeg(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder  = decoder
    def forward(self, x):
        # I-JEPA models return patch tokens directly
        patches = self.backbone.forward_features(x)   # (B, N, D)
        B, N, D = patches.shape
        h = w = int(math.sqrt(N))                     # 28 for 448 px / 16
        patches = patches.view(B, h, w, D).permute(0, 3, 1, 2)
        logits  = self.decoder(patches)               # (B, num_cls, 28, 28)
        return logits

model = JepaSeg(backbone, Decoder(1280, cfg['num_classes'])).to(device)

# %% [markdown]
# # 6.  Loss & optimiser

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
# %% [markdown]
# # 7.  Metrics

# %%
def iou_pytorch(pred, target, num_classes=2, eps=1e-6):
    # pred: (B,H,W) after argmax
    ious = []
    for cls in range(num_classes):
        intersect = ((pred == cls) & (target == cls)).sum((1,2)).float()
        union = ((pred == cls) | (target == cls)).sum((1,2)).float()
        iou = (intersect + eps) / (union + eps)
        ious.append(iou.mean().item())
    return np.mean(ious)

# %% [markdown]
# # 8.  Training / validation loop

# %%
writer = SummaryWriter('runs/kvasir_ijepa_seg')     # logs go here
print('TensorBoard command:  tensorboard --logdir=runs')

best_iou = 0.
for epoch in range(1, cfg['epochs']+1):
    # ---- train ----
    model.train()
    backbone.eval()
    train_loss, train_iou = [], []
    for img, mask in tqdm(train_loader, leave=False):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(img)
        logits = nn.functional.interpolate(logits, size=mask.shape[-2:],
                                           mode='bilinear', align_corners=False)
        loss = criterion(logits, mask)
        loss.backward()

        # <== TB  log gradient norm
        total_norm = 0.
        for p in model.decoder.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()

        train_loss.append(loss.item())
        pred = logits.argmax(1)
        train_iou.append(iou_pytorch(pred, mask))

    # ---- val ----
    model.eval()
    val_loss, val_iou = [], []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            logits = nn.functional.interpolate(logits, size=mask.shape[-2:],
                                               mode='bilinear', align_corners=False)
            val_loss.append(criterion(logits, mask).item())
            val_iou.append(iou_pytorch(logits.argmax(1), mask))

    # ---- aggregate ----
    tr_loss, tr_miou = np.mean(train_loss), np.mean(train_iou)
    vl_loss, vl_miou = np.mean(val_loss),   np.mean(val_iou)

    # <== TB  write scalars
    writer.add_scalar('Loss/train', tr_loss, epoch)
    writer.add_scalar('mIoU/train', tr_miou, epoch)
    writer.add_scalar('Loss/val',   vl_loss, epoch)
    writer.add_scalar('mIoU/val',   vl_miou, epoch)
    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('GradNorm/decoder', grad_norm, epoch)

    # console print unchanged
    print(f'E{epoch:02d} | '
          f'train loss {tr_loss:.4f} mIoU {tr_miou:.4f} | '
          f'val loss {vl_loss:.4f} mIoU {vl_miou:.4f}')
    if vl_miou > best_iou:
        best_iou = vl_miou
        torch.save(model.state_dict(), 'weights/best_jepa_seg_kvasir.pth')
        print('  ↑ best model saved.')

# <== TB  close writer when done
writer.close()
# %% [markdown]
# # 9.  Inference helper

# %%

def infer(image_path, weight_path='weights/best_jepa_seg_kvasir.pth'):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug = val_aug(image=img)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(aug)
        mask = logits.argmax(1).squeeze(0).cpu().numpy()
    return mask

# Visualise
mask = infer(r'kvasir-sessile\sessile-main-Kvasir-SEG\images\cju0qoxqj9q6s0835b43399p4.jpg')
import matplotlib.pyplot as plt
plt.imshow(mask); plt.axis('off'); plt.show()



