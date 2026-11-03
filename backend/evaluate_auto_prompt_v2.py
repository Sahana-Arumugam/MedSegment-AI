from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "backend" / "prepared_dataset" / "busi_manifest.csv"
CHECKPOINT = ROOT / "backend" / "models" / "auto_prompt_cnn_v2.pth"
OUTPUT_DIR = ROOT / "backend" / "evaluation_examples_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_box(row):
    width, height = float(row["width"]), float(row["height"])
    x1 = float(row["bbox_x1"]) / width
    y1 = float(row["bbox_y1"]) / height
    x2 = float(row["bbox_x2"]) / width
    y2 = float(row["bbox_y2"]) / height
    return np.array([
        (x1 + x2) / 2, (y1 + y2) / 2,
        max(x2 - x1, 1e-4), max(y2 - y1, 1e-4)
    ], dtype=np.float32)


class BUSITestDataset(Dataset):
    def __init__(self, records, image_size):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise RuntimeError(f"Cannot read image: {row['image_path']}")
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        target = torch.from_numpy(parse_box(row))
        return image, target, index, w, h


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x), inplace=True)


class AutoPromptCNNV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(3, 24, stride=2),
            ConvBlock(24, 32, stride=2),
            ConvBlock(32, 48, stride=2),
            ConvBlock(48, 64, stride=2),
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


def calculate_iou(pred, target):
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


def main():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE,
                            weights_only=False)
    image_size = int(checkpoint["image_size"])

    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = [
        r for r in rows
        if r["split"] == "test"
        and r["category"] in ("benign", "malignant")
        and r["has_bbox"].lower() == "true"
    ]
    if not records:
        raise RuntimeError("No annotated test records found.")

    loader = DataLoader(
        BUSITestDataset(records, image_size),
        batch_size=8, shuffle=False, num_workers=0
    )
    model = AutoPromptCNNV2().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_ious, examples = [], []
    with torch.no_grad():
        for images, targets, indices, widths, heights in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            predictions = model(images)
            ious = calculate_iou(predictions, targets)
            all_ious.extend(ious.cpu().tolist())

            pred_boxes = to_xyxy(predictions).cpu().numpy()
            true_boxes = to_xyxy(targets).cpu().numpy()
            for j, idx in enumerate(indices.tolist()):
                examples.append((
                    float(ious[j].item()), idx,
                    pred_boxes[j], true_boxes[j]
                ))

    values = np.asarray(all_ious)
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint best validation box IoU:",
          checkpoint.get("best_val_box_iou"))
    print("Test images:", len(values))
    print(f"Mean box IoU:   {values.mean():.4f}")
    print(f"Median box IoU: {np.median(values):.4f}")
    print(f"IoU >= 0.50:    {(values >= 0.50).mean():.1%}")
    print(f"IoU >= 0.75:    {(values >= 0.75).mean():.1%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples.sort(key=lambda item: item[0])
    selected = examples[:5] + examples[-5:]

    for rank, (iou, idx, pred, truth) in enumerate(selected, 1):
        row = records[idx]
        image = cv2.imread(row["image_path"])
        h, w = image.shape[:2]

        def pixel_box(box):
            x1, y1, x2, y2 = box
            return (
                int(round(x1 * w)), int(round(y1 * h)),
                int(round(x2 * w)), int(round(y2 * h))
            )

        cv2.rectangle(image, pixel_box(truth), (0, 255, 0), 2)
        cv2.rectangle(image, pixel_box(pred), (0, 0, 255), 2)
        cv2.putText(image, f"IoU: {iou:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        output = OUTPUT_DIR / f"example_{rank:02d}.png"
        cv2.imwrite(str(output), image)

    print("Saved visual examples to:", OUTPUT_DIR)
    print("Green = ground truth; Red = prediction.")


if __name__ == "__main__":
    main()
