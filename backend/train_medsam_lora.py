
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
MEDSAM_DIR = ROOT / "MedSAM"
MANIFEST = ROOT / "backend" / "prepared_dataset" / "busi_manifest.csv"
CHECKPOINT = MEDSAM_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
OUTPUT = ROOT / "backend" / "models" / "medsam_lora_experiment.pth"

sys.path.insert(0, str(MEDSAM_DIR))
from segment_anything import sam_model_registry

torch.manual_seed(42)
DEVICE = torch.device("cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 1
EPOCHS = 1
MAX_TRAIN_IMAGES = 12
LR = 1e-3

class BUSIDataset(Dataset):
    def __init__(self, manifest, split, limit=None):
        df = pd.read_csv(manifest)
        df = df[
            (df["split"].astype(str).str.lower() == split)
            & (df["has_mask"].astype(str).str.lower().isin(["true", "1"]))
        ].copy()

        if limit:
            df = df.head(limit)

        self.rows = df.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        image = Image.open(image_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask_paths = str(row["mask_paths"]).split(";")
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        for value in mask_paths:
            value = value.strip()
            if not value or value.lower() == "nan":
                continue

            mask_path = Path(value)
            if not mask_path.is_absolute():
                mask_path = ROOT / mask_path

            if mask_path.exists():
                m = Image.open(mask_path).convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.NEAREST,
                )
                mask = np.maximum(mask, np.asarray(m))

        mask = (mask > 0).astype(np.float32)
        return image, torch.from_numpy(mask).unsqueeze(0)

class LoRAConv(nn.Module):
    """Low-rank trainable residual adapter for a convolution feature map."""
    def __init__(self, channels, rank=4):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0

    def forward(self, x):
        return x + self.scale * self.up(self.down(x))

class MedSAMWithLoRA(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.adapter = LoRAConv(256, rank=4)

    def forward(self, images, masks):
        # This lightweight experiment trains a low-rank feature adapter
        # and segmentation head on frozen pretrained image features.
        with torch.no_grad():
            features = self.sam.image_encoder(images)

        features = self.adapter(features)
        features = nn.functional.interpolate(
            features,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # A small segmentation head maps frozen features to a mask.
        logits = self.head(features)
        return logits

    def build_head(self):
        self.head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {CHECKPOINT}")

    print("Loading pretrained MedSAM ViT-B on CPU...")
    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
    sam.to(DEVICE)
    sam.eval()

    # Freeze original pretrained model weights.
    for param in sam.parameters():
        param.requires_grad = False

    model = MedSAMWithLoRA(sam)
    model.build_head()
    model.to(DEVICE)

    dataset = BUSIDataset(MANIFEST, "train", MAX_TRAIN_IMAGES)
    if len(dataset) == 0:
        raise RuntimeError("No annotated training images found in manifest.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(
        list(model.adapter.parameters()) + list(model.head.parameters()),
        lr=LR,
    )
    criterion = nn.BCEWithLogitsLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training images: {len(dataset)}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print("Running one CPU epoch...")

    start = time.time()
    model.train()
    losses = []

    for step, (images, masks) in enumerate(loader, 1):
        # SAM image encoder expects 1024x1024 RGB-normalized input.
        images = nn.functional.interpolate(
            images, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        images = images * 255.0
        images = (images - mean) / std
        masks = nn.functional.interpolate(
            masks, size=(64, 64), mode="nearest"
        )

        optimizer.zero_grad()
        logits = model(images, masks)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        print(f"Step {step}/{len(loader)} | loss={loss.item():.4f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": model.adapter.state_dict(),
            "head": model.head.state_dict(),
            "epochs": EPOCHS,
            "train_images": len(dataset),
            "mean_train_loss": float(np.mean(losses)),
            "device": str(DEVICE),
            "note": (
                "Experimental low-rank feature adapter and segmentation head; "
                "not a PEFT LoRA adapter injected into MedSAM attention layers."
            ),
        },
        OUTPUT,
    )

    print(f"Mean training loss: {np.mean(losses):.4f}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")
    print(f"Saved experiment checkpoint: {OUTPUT}")
    print("This is a prototype experiment, not evidence of improved MedSAM performance.")

if __name__ == "__main__":
    main()
