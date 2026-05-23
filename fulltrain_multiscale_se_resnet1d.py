# -*- coding: utf-8 -*-
import os
import glob
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


# =========================================================
# 0. Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)


# =========================================================
# 1. Config
# =========================================================
ROOT = "./DM_data3"
TRAIN_ROOT = os.path.join(ROOT, "train")
TEST_ROOT = os.path.join(ROOT, "test")

FEATURE_COLS = [
    "mean_x", "mean_y", "mean_z",
    "std_x", "std_y", "std_z"
]

NUM_CLASSES = 6

BATCH_SIZE = 32
NUM_WORKERS = 0

LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30

# 原本 baseline augmentation
GAUSSIAN_NOISE_STD = 0.02
TIME_MASK_PROB = 0.5
TIME_MASK_MAX_WIDTH = 20
TIME_SHIFT_PROB = 0.5
TIME_SHIFT_MAX = 15

USE_TTA = True

# 固定 epoch：建議先跑 25, 30, 35
EPOCH_LIST = [38]

USE_SWA = True
SWA_START = 20
SWA_LR = 3e-4

SAVE_DIR = "./fulltrain_multiscale_se_resnet1d_baseline3"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================================================
# 2. File collection
# =========================================================
def collect_all_train_files(train_root):
    train_files = []
    user_dirs = sorted(glob.glob(os.path.join(train_root, "User_*")))

    for user_dir in user_dirs:
        train_files.extend(sorted(glob.glob(os.path.join(user_dir, "*.csv"))))

    return train_files


def collect_test_files(test_root):
    test_files = []
    user_dirs = sorted(glob.glob(os.path.join(test_root, "User_*")))

    for user_dir in user_dirs:
        test_files.extend(sorted(glob.glob(os.path.join(user_dir, "*.csv"))))

    return test_files


# =========================================================
# 3. Normalization
# =========================================================
def compute_feature_stats(file_list, feature_cols):
    all_sum = np.zeros(len(feature_cols), dtype=np.float64)
    all_sq_sum = np.zeros(len(feature_cols), dtype=np.float64)
    total_count = 0

    for path in file_list:
        df = pd.read_csv(path)
        x = df[feature_cols].values.astype(np.float64)

        all_sum += x.sum(axis=0)
        all_sq_sum += (x ** 2).sum(axis=0)
        total_count += x.shape[0]

    mean = all_sum / total_count
    var = all_sq_sum / total_count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    return mean.astype(np.float32), std.astype(np.float32)


# =========================================================
# 4. Augmentation
# =========================================================
def apply_time_mask(x, max_width=20):
    x = x.copy()
    T = x.shape[0]

    if T <= 1:
        return x

    width = np.random.randint(5, min(max_width, T) + 1)
    start = np.random.randint(0, T - width + 1)

    x[start:start + width, :] = 0.0
    return x


def apply_time_shift(x, max_shift=15):
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift=shift, axis=0).copy()


def apply_gaussian_noise(x, std=0.02):
    noise = np.random.normal(0, std, size=x.shape).astype(np.float32)
    return x + noise


# =========================================================
# 5. Dataset
# =========================================================
class HARFileDataset(Dataset):
    def __init__(
        self,
        file_list,
        feature_cols,
        mean=None,
        std=None,
        is_test=False,
        augment=False
    ):
        self.file_list = file_list
        self.feature_cols = feature_cols
        self.mean = mean
        self.std = std
        self.is_test = is_test
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        df = pd.read_csv(path)

        x = df[self.feature_cols].values.astype(np.float32)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        if self.augment and not self.is_test:
            if np.random.rand() < TIME_SHIFT_PROB:
                x = apply_time_shift(x, max_shift=TIME_SHIFT_MAX)

            if np.random.rand() < TIME_MASK_PROB:
                x = apply_time_mask(x, max_width=TIME_MASK_MAX_WIDTH)

            x = apply_gaussian_noise(x, std=GAUSSIAN_NOISE_STD)

        x = torch.tensor(x, dtype=torch.float32)

        file_id = int(df["file_id"].iloc[0])

        if self.is_test:
            return x, file_id

        y = int(df["label"].iloc[0])
        return x, torch.tensor(y, dtype=torch.long)


# =========================================================
# 6. Model Blocks: Original Baseline
# =========================================================
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.30):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        return out


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.shape

        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)

        return x * w


class MultiScaleSEBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.30,
        reduction=8
    ):
        super().__init__()

        branch_channels = out_channels // 3
        last_branch_channels = out_channels - branch_channels * 2

        self.branch3 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch7 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                last_branch_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(last_branch_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.conv_out = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(out_channels)
        )

        self.se = SEBlock1D(out_channels, reduction=reduction)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)

        out = torch.cat([b3, b5, b7], dim=1)
        out = self.fuse(out)
        out = self.conv_out(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        return out


class HybridMultiScaleSEResNet1DClassifier(nn.Module):
    def __init__(self, input_dim=6, num_classes=6, dropout=0.30):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_dim,
                64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            BasicBlock1D(64, 64, kernel_size=7, stride=1, dropout=dropout),
            BasicBlock1D(64, 64, kernel_size=7, stride=1, dropout=dropout)
        )

        self.layer2 = nn.Sequential(
            BasicBlock1D(64, 128, kernel_size=7, stride=2, dropout=dropout),
            BasicBlock1D(128, 128, kernel_size=7, stride=1, dropout=dropout)
        )

        self.layer3 = nn.Sequential(
            MultiScaleSEBlock1D(128, 256, stride=2, dropout=dropout),
            MultiScaleSEBlock1D(256, 256, stride=1, dropout=dropout)
        )

        self.layer4 = nn.Sequential(
            MultiScaleSEBlock1D(256, 256, stride=1, dropout=dropout),
            MultiScaleSEBlock1D(256, 256, stride=1, dropout=dropout)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C] = [B, 300, 6]
        x = x.transpose(1, 2)  # [B, 6, 300]

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).squeeze(-1)
        logits = self.head(x)

        return logits


# =========================================================
# 7. TTA
# =========================================================
def predict_tta(model, x):
    logits_list = []

    # 原始輸入加權
    logits_list.append(model(x) * 2.0)

    for shift in [-15, -10, -5, 5, 10, 15]:
        logits_list.append(model(torch.roll(x, shifts=shift, dims=1)))

    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return logits


# =========================================================
# 8. Train full data to fixed epoch
# =========================================================
def train_full_model(target_epoch, train_loader):
    print("\n" + "=" * 80)
    print(f"Full Train Model | Target Epoch = {target_epoch}")
    print("=" * 80)

    set_seed(SEED)

    model = HybridMultiScaleSEResNet1DClassifier(
        input_dim=len(FEATURE_COLS),
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=target_epoch
    )

    if USE_SWA:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=SWA_LR
        )

    for epoch in range(target_epoch):
        model.train()

        total_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            correct += (preds == y).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        if USE_SWA and (epoch + 1) >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        print(
            f"[Epoch {epoch+1:02d}/{target_epoch}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        )

    if USE_SWA and target_epoch >= SWA_START:
        print("Updating BN for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        final_model = swa_model.module
    else:
        final_model = model

    return final_model


# =========================================================
# 9. Inference
# =========================================================
def predict_test(model, test_loader):
    model.eval()

    results = []

    with torch.no_grad():
        for x, file_ids in test_loader:
            x = x.to(device)

            if USE_TTA:
                logits = predict_tta(model, x)
            else:
                logits = model(x)

            preds = logits.argmax(dim=1).cpu().numpy()

            for fid, pred in zip(file_ids, preds):
                results.append([int(fid), int(pred)])

    return results


def save_submission(results, output_csv):
    sub = pd.DataFrame(results, columns=["Id", "Label"])
    sub = sub.sort_values("Id").reset_index(drop=True)
    sub.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print(sub.head())


# =========================================================
# 10. Main
# =========================================================
train_files = collect_all_train_files(TRAIN_ROOT)
test_files = collect_test_files(TEST_ROOT)

print("Train files:", len(train_files))
print("Test files :", len(test_files))

mean, std = compute_feature_stats(train_files, FEATURE_COLS)

print("Feature mean:", mean)
print("Feature std :", std)

train_dataset = HARFileDataset(
    train_files,
    feature_cols=FEATURE_COLS,
    mean=mean,
    std=std,
    is_test=False,
    augment=True
)

test_dataset = HARFileDataset(
    test_files,
    feature_cols=FEATURE_COLS,
    mean=mean,
    std=std,
    is_test=True,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


for target_epoch in EPOCH_LIST:
    model = train_full_model(
        target_epoch=target_epoch,
        train_loader=train_loader
    )

    results = predict_test(
        model=model,
        test_loader=test_loader
    )

    output_csv = os.path.join(
        SAVE_DIR,
        f"submission_fulltrain_epoch{target_epoch}_multiscale_se_resnet1d_6feat_tta.csv"
    )

    save_submission(results, output_csv)

print("\nAll full-train submissions done.")