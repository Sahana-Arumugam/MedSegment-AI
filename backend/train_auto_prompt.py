
from pathlib import Path
import csv
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "backend" / "prepared_dataset" / "busi_manifest.csv"
OUTPUT = ROOT / "backend" / "models" / "auto_prompt_cnn.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", DEVICE)


class BUSIBBoxDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Cannot read image: {row['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))

        width = float(row["width"])
        height = float(row["height"])

        x1 = float(row["bbox_x1"]) / width
        y1 = float(row["bbox_y1"]) / height
        x2 = float(row["bbox_x2"]) / width
        y2 = float(row["bbox_y2"]) / height

        # Center x, center y, width, height
        target = torch.tensor(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
            dtype=torch.float32,
        )

        return image, target


class AutoPromptCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def center_to_xyxy(box):
    cx, cy, w, h = box.unbind(dim=1)
    return torch.stack([
        (cx - w / 2).clamp(0, 1),
        (cy - h / 2).clamp(0, 1),
        (cx + w / 2).clamp(0, 1),
        (cy + h / 2).clamp(0, 1),
    ], dim=1)


def box_iou(pred, target):
    p = center_to_xyxy(pred)
    t = center_to_xyxy(target)

    ix1 = torch.maximum(p[:, 0], t[:, 0])
    iy1 = torch.maximum(p[:, 1], t[:, 1])
    ix2 = torch.minimum(p[:, 2], t[:, 2])
    iy2 = torch.minimum(p[:, 3], t[:, 3])

    intersection = (ix2 - ix1).clamp(min=0) * (
        iy2 - iy1
    ).clamp(min=0)

    p_area = (p[:, 2] - p[:, 0]).clamp(min=0) * (
        p[:, 3] - p[:, 1]
    ).clamp(min=0)

    t_area = (t[:, 2] - t[:, 0]).clamp(min=0) * (
        t[:, 3] - t[:, 1]
    ).clamp(min=0)

    return intersection / (p_area + t_area - intersection + 1e-7)


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

    print(f"Annotated images: {len(records)}")
    print(f"Training images: {len(train)}")
    print(f"Validation images: {len(val)}")

    return train, val


def main():
    train_records, val_records = load_records()

    train_loader = DataLoader(
        BUSIBBoxDataset(train_records),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        BUSIBBoxDataset(val_records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AutoPromptCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )
    criterion = nn.SmoothL1Loss()

    best_iou = -1.0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        val_ious = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * images.size(0)

                val_ious.extend(
                    box_iou(predictions, targets).cpu().tolist()
                )

        train_loss /= len(train_records)
        val_loss /= len(val_records)
        mean_iou = float(np.mean(val_ious))

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val box IoU: {mean_iou:.4f}"
        )

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "best_val_box_iou": best_iou,
                    "epoch": epoch + 1,
                },
                OUTPUT,
            )

    print(f"\nBest validation box IoU: {best_iou:.4f}")
    print(f"Model saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
