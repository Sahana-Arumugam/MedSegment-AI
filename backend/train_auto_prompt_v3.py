from pathlib import Path
import csv
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "backend" / "prepared_dataset" / "busi_manifest.csv"
OUTPUT = ROOT / "backend" / "models" / "auto_prompt_cnn_v3.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", DEVICE)


def parse_box(row):
    width, height = float(row["width"]), float(row["height"])
    x1 = float(row["bbox_x1"]) / width
    y1 = float(row["bbox_y1"]) / height
    x2 = float(row["bbox_x2"]) / width
    y2 = float(row["bbox_y2"]) / height
    # Target is normalized center-x, center-y, width, height.
    return np.array([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        max(x2 - x1, 1e-4),
        max(y2 - y1, 1e-4),
    ], dtype=np.float32)


class BUSIBBoxDataset(Dataset):
    def __init__(self, records, training=False):
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {row['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = parse_box(row)

        # Horizontal flip: mirror the target center too.
        if self.training and random.random() < 0.5:
            image = cv2.flip(image, 1)
            target[0] = 1.0 - target[0]

        # Mild brightness/contrast augmentation only; do not geometrically
        # distort the image because that would require transforming the box.
        if self.training and random.random() < 0.35:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-10, 10)
            image = np.clip(image.astype(np.float32) * alpha + beta,
                            0, 255).astype(np.uint8)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE),
                           interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        target = torch.from_numpy(target.copy())
        return image, target


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x), inplace=True)


class AutoPromptCNNV2(nn.Module):
    """Spatially aware lightweight CNN for normalized bounding-box regression."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 24, stride=2),      # 128 x 128
            ConvBlock(24, 32, stride=2),     # 64 x 64
            ConvBlock(32, 48, stride=2),     # 32 x 32
            ConvBlock(48, 64, stride=2),     # 16 x 16
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def to_xyxy(box):
    cx, cy, w, h = box.unbind(dim=1)
    return torch.stack([
        (cx - w / 2).clamp(0, 1),
        (cy - h / 2).clamp(0, 1),
        (cx + w / 2).clamp(0, 1),
        (cy + h / 2).clamp(0, 1),
    ], dim=1)


def box_iou(pred, target):
    p, t = to_xyxy(pred), to_xyxy(target)
    ix1 = torch.maximum(p[:, 0], t[:, 0])
    iy1 = torch.maximum(p[:, 1], t[:, 1])
    ix2 = torch.minimum(p[:, 2], t[:, 2])
    iy2 = torch.minimum(p[:, 3], t[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pa = (p[:, 2] - p[:, 0]).clamp(min=0) * (
        p[:, 3] - p[:, 1]).clamp(min=0)
    ta = (t[:, 2] - t[:, 0]).clamp(min=0) * (
        t[:, 3] - t[:, 1]).clamp(min=0)
    return inter / (pa + ta - inter + 1e-7)


def giou_loss(pred, target):
    p, t = to_xyxy(pred), to_xyxy(target)
    ix1 = torch.maximum(p[:, 0], t[:, 0])
    iy1 = torch.maximum(p[:, 1], t[:, 1])
    ix2 = torch.minimum(p[:, 2], t[:, 2])
    iy2 = torch.minimum(p[:, 3], t[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pa = (p[:, 2] - p[:, 0]).clamp(min=0) * (
        p[:, 3] - p[:, 1]).clamp(min=0)
    ta = (t[:, 2] - t[:, 0]).clamp(min=0) * (
        t[:, 3] - t[:, 1]).clamp(min=0)
    union = pa + ta - inter
    iou = inter / (union + 1e-7)

    cx1 = torch.minimum(p[:, 0], t[:, 0])
    cy1 = torch.minimum(p[:, 1], t[:, 1])
    cx2 = torch.maximum(p[:, 2], t[:, 2])
    cy2 = torch.maximum(p[:, 3], t[:, 3])
    enclosing = (cx2 - cx1).clamp(min=0) * (
        cy2 - cy1).clamp(min=0)
    giou = iou - (enclosing - union) / (enclosing + 1e-7)
    return (1.0 - giou).mean()


def load_records():
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    records = [
        r for r in records
        if r["category"] in ("benign", "malignant")
        and r["has_bbox"].lower() == "true"
    ]
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]

    if not train or not val:
        raise RuntimeError("Training or validation records are missing.")

    print(f"Annotated records: {len(records)}")
    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(val)}")
    print("Note: test split is not used for model selection.")
    return train, val



def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, all_ious, count = 0.0, [], 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.set_grad_enabled(training):
            predictions = model(images)

            # Per-image coordinate regression loss.
            coordinate_loss = F.smooth_l1_loss(
                predictions, targets, reduction="none"
            ).mean(dim=1)

            # Per-image GIoU loss.
            giou_losses = []
            for i in range(images.size(0)):
                giou_losses.append(
                    giou_loss(
                        predictions[i:i+1],
                        targets[i:i+1]
                    )
                )
            giou_losses = torch.stack(giou_losses)

            # Ground-truth box area is available during training.
            # Clamp the weights to avoid excessive influence from tiny boxes.
            target_area = (
                targets[:, 2].clamp(min=1e-4)
                * targets[:, 3].clamp(min=1e-4)
            )
            weights = torch.rsqrt(target_area)
            weights = weights.clamp(min=1.0, max=3.0)
            weights = weights / weights.mean().detach()

            per_image_loss = coordinate_loss + 0.5 * giou_losses
            loss = (weights * per_image_loss).mean()

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch = images.size(0)
        total_loss += loss.item() * batch
        all_ious.extend(
            box_iou(predictions.detach(), targets).cpu().tolist()
        )
        count += batch

    return total_loss / max(count, 1), float(np.mean(all_ious))



def main():
    train_records, val_records = load_records()
    train_loader = DataLoader(
        BUSIBBoxDataset(train_records, training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        BUSIBBoxDataset(val_records, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = AutoPromptCNNV2().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    best_iou, best_epoch, stale_epochs = -1.0, 0, 0
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader)
        scheduler.step(val_iou)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss {train_loss:.4f} | train IoU {train_iou:.4f} | "
            f"val loss {val_loss:.4f} | val box IoU {val_iou:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_iou > best_iou:
            best_iou, best_epoch, stale_epochs = val_iou, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "best_val_box_iou": best_iou,
                "epoch": best_epoch,
                "architecture": "AutoPromptCNNV2",
            }, OUTPUT)
            print(f"  Saved best checkpoint: {OUTPUT}")
        else:
            stale_epochs += 1

        if stale_epochs >= PATIENCE:
            print(f"Early stopping after {epoch} epochs.")
            break

    print(f"\nBest validation box IoU: {best_iou:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {OUTPUT}")
    print(f"Elapsed minutes: {(time.time() - start) / 60:.1f}")
    print("Existing auto_prompt_cnn.pth was not changed.")


if __name__ == "__main__":
    main()
