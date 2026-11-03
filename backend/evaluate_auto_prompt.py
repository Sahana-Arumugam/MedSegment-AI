
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
CHECKPOINT = ROOT / "backend" / "models" / "auto_prompt_cnn.pth"
OUTPUT_DIR = ROOT / "backend" / "evaluation_examples"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BUSIBBoxDataset(Dataset):
    def __init__(self, records, image_size):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = cv2.imread(row["image_path"])

        if image is None:
            raise RuntimeError(f"Cannot read {row['image_path']}")

        original_h, original_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.image_size, self.image_size)
        )
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))

        w = float(row["width"])
        h = float(row["height"])
        x1 = float(row["bbox_x1"]) / w
        y1 = float(row["bbox_y1"]) / h
        x2 = float(row["bbox_x2"]) / w
        y2 = float(row["bbox_y2"]) / h

        target = torch.tensor(
            [(x1 + x2) / 2, (y1 + y2) / 2,
             x2 - x1, y2 - y1],
            dtype=torch.float32,
        )

        return image, target, index, original_w, original_h


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


def to_xyxy(box):
    cx, cy, w, h = box.unbind(dim=1)
    return torch.stack([
        (cx - w / 2).clamp(0, 1),
        (cy - h / 2).clamp(0, 1),
        (cx + w / 2).clamp(0, 1),
        (cy + h / 2).clamp(0, 1),
    ], dim=1)


def calculate_iou(pred, target):
    p = to_xyxy(pred)
    t = to_xyxy(target)

    ix1 = torch.maximum(p[:, 0], t[:, 0])
    iy1 = torch.maximum(p[:, 1], t[:, 1])
    ix2 = torch.minimum(p[:, 2], t[:, 2])
    iy2 = torch.minimum(p[:, 3], t[:, 3])

    intersection = (
        (ix2 - ix1).clamp(min=0)
        * (iy2 - iy1).clamp(min=0)
    )
    pa = (
        (p[:, 2] - p[:, 0]).clamp(min=0)
        * (p[:, 3] - p[:, 1]).clamp(min=0)
    )
    ta = (
        (t[:, 2] - t[:, 0]).clamp(min=0)
        * (t[:, 3] - t[:, 1]).clamp(min=0)
    )
    return intersection / (pa + ta - intersection + 1e-7)


def main():
    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    image_size = checkpoint["image_size"]

    with MANIFEST.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    test_records = [
        r for r in rows
        if r["split"] == "test"
        and r["category"] in ("benign", "malignant")
        and r["has_bbox"].lower() == "true"
    ]

    if not test_records:
        raise RuntimeError("No annotated test records found.")

    dataset = BUSIBBoxDataset(test_records, image_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = AutoPromptCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_ious = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = []

    with torch.no_grad():
        for images, targets, indices, widths, heights in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
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
    print("Test images:", len(values))
    print(f"Mean box IoU:   {values.mean():.4f}")
    print(f"Median box IoU: {np.median(values):.4f}")
    print(f"IoU >= 0.50:    {(values >= 0.50).mean():.1%}")
    print(f"IoU >= 0.75:    {(values >= 0.75).mean():.1%}")

    # Save the five worst and five best examples.
    examples.sort(key=lambda x: x[0])
    selected = examples[:5] + examples[-5:]

    for rank, (iou, idx, pred, truth) in enumerate(selected, 1):
        row = test_records[idx]
        image = cv2.imread(row["image_path"])
        h, w = image.shape[:2]

        def pixel_box(box):
            x1, y1, x2, y2 = box
            return (
                int(round(x1 * w)), int(round(y1 * h)),
                int(round(x2 * w)), int(round(y2 * h))
            )

        cv2.rectangle(
            image, pixel_box(truth), (0, 255, 0), 2
        )
        cv2.rectangle(
            image, pixel_box(pred), (0, 0, 255), 2
        )
        cv2.putText(
            image, f"IoU: {iou:.3f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        filename = OUTPUT_DIR / f"example_{rank:02d}.png"
        cv2.imwrite(str(filename), image)
        print(filename)

    print("Green = ground truth; Red = prediction.")


if __name__ == "__main__":
    main()
