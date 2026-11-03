from pathlib import Path
import sys
import csv
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
MANIFEST_PATH = BACKEND_DIR / "prepared_dataset" / "busi_manifest.csv"
PROMPT_CHECKPOINT = BACKEND_DIR / "models" / "auto_prompt_cnn_v2.pth"
PROMPT_CHECKPOINT_V3 = BACKEND_DIR / "models" / "auto_prompt_cnn_v3.pth"
MEDSAM_DIR = PROJECT_DIR / "MedSAM"
MEDSAM_CHECKPOINT = MEDSAM_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"

IMAGE_SIZE = 256
MEDSAM_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = BACKEND_DIR / "box_coordinate_diagnostics"
VISUAL_DIR = OUTPUT_DIR / "visual_examples"
OUTPUT_CSV = OUTPUT_DIR / "box_coordinate_metrics.csv"
# ------------------------------------------------------------
# Exact AutoPromptCNNV2 architecture used for training
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if in_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return torch.relu(self.block(x) + self.skip(x))


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


# ------------------------------------------------------------
# Metrics and mask helpers
# ------------------------------------------------------------
def calculate_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    dice_denom = pred_count + gt_count
    dice = (2.0 * tp / dice_denom) if dice_denom else 1.0
    iou = (tp / union) if union else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0

    return {
        "dice": dice,
        "segmentation_iou": iou,
        "precision": precision,
        "recall": recall,
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "predicted_foreground_pixels": pred_count,
        "ground_truth_foreground_pixels": gt_count,
    }


def load_ground_truth_mask(mask_paths, image_shape):
    """Combine all available masks for one image into a single binary mask."""
    height, width = image_shape
    combined = np.zeros((height, width), dtype=bool)
    found = 0

    for mask_path in mask_paths:
        mask_path = Path(str(mask_path).strip())
        if not str(mask_path) or not mask_path.is_file():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )

        combined |= mask > 0
        found += 1

    return combined, found


def bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def save_visual(image_rgb, gt_mask, pred_mask, pred_box, out_path):
    """Save image with GT contour (cyan), predicted mask (red), predicted box (green)."""
    canvas = image_rgb.copy()
    overlay = canvas.copy()
    overlay[pred_mask] = (
        0.65 * overlay[pred_mask] + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)
    canvas = overlay

    contours_gt, _ = cv2.findContours(
        gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Convert RGB to BGR for OpenCV drawing, then back to RGB for writing.
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours_gt, -1, (255, 255, 0), 2)  # cyan in RGB
    x1, y1, x2, y2 = [int(round(v)) for v in pred_box]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(out_path), bgr)


def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not PROMPT_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Prompt checkpoint not found: {PROMPT_CHECKPOINT}")
    if not MEDSAM_CHECKPOINT.is_file():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {MEDSAM_CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)
    required = {
        "image_path", "mask_paths", "bbox_x1", "bbox_y1",
        "bbox_x2", "bbox_y2", "has_mask", "has_bbox", "split"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    # Evaluate the established held-out test split only.
    test_df = df[
        (df["split"].astype(str).str.lower() == "test")
        & (df["has_mask"].astype(str).str.lower() == "true")
        & (df["has_bbox"].astype(str).str.lower() == "true")
    ].copy()

    print(f"Device: {DEVICE}")
    print(f"Manifest records: {len(df)}")
    print(f"Annotated test images selected: {len(test_df)}")
    if len(test_df) == 0:
        raise RuntimeError("No annotated test images found in the manifest.")

    if str(MEDSAM_DIR) not in sys.path:
        sys.path.insert(0, str(MEDSAM_DIR))
    from segment_anything import sam_model_registry, SamPredictor

    if not PROMPT_CHECKPOINT.is_file():
        raise FileNotFoundError(f"V2 checkpoint not found: {PROMPT_CHECKPOINT}")
    if not PROMPT_CHECKPOINT_V3.is_file():
        raise FileNotFoundError(f"V3 checkpoint not found: {PROMPT_CHECKPOINT_V3}")

    prompt_model_v2 = AutoPromptCNNV2().to(DEVICE)
    checkpoint_v2 = torch.load(
        PROMPT_CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    if "model_state_dict" not in checkpoint_v2:
        raise RuntimeError("V2 checkpoint lacks 'model_state_dict'.")
    prompt_model_v2.load_state_dict(checkpoint_v2["model_state_dict"])
    prompt_model_v2.eval()

    prompt_model_v3 = AutoPromptCNNV2().to(DEVICE)
    checkpoint_v3 = torch.load(
        PROMPT_CHECKPOINT_V3, map_location=DEVICE, weights_only=False
    )
    if "model_state_dict" not in checkpoint_v3:
        raise RuntimeError("V3 checkpoint lacks 'model_state_dict'.")
    prompt_model_v3.load_state_dict(checkpoint_v3["model_state_dict"])
    prompt_model_v3.eval()

    sam = sam_model_registry["vit_b"](checkpoint=str(MEDSAM_CHECKPOINT))
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)

    
    results = []
    started = time.time()

    model_configs = [
        ("V2", prompt_model_v2),
        ("V3", prompt_model_v3),
    ]

    for index, (_, row) in enumerate(test_df.iterrows(), start=1):
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            print(f"[SKIP] Image missing: {image_path}")
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[SKIP] Could not decode: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        raw_mask_paths = str(row["mask_paths"]).strip()
        mask_paths = [
            item.strip() for item in raw_mask_paths.split(";")
            if item.strip() and item.strip().lower() != "nan"
        ]
        gt_mask, mask_count = load_ground_truth_mask(
            mask_paths, (height, width)
        )
        if mask_count == 0 or not gt_mask.any():
            print(f"[SKIP] No usable non-empty GT mask: {image_path.name}")
            continue

        ys, xs = np.where(gt_mask)
        gt_box = [
            float(xs.min()), float(ys.min()),
            float(xs.max() + 1), float(ys.max() + 1)
        ]

        # Define lesion size using the ground-truth mask area fraction.
        area_fraction = float(gt_mask.sum()) / float(height * width)
        if area_fraction < 0.05:
            lesion_size = "small"
        elif area_fraction < 0.15:
            lesion_size = "medium"
        else:
            lesion_size = "large"

        prompt_image = cv2.resize(
            image_rgb, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0
        tensor = torch.from_numpy(
            prompt_image.transpose(2, 0, 1).copy()
        ).unsqueeze(0).to(DEVICE)

        medsam_image = cv2.resize(
            image_rgb, (MEDSAM_SIZE, MEDSAM_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # Compute MedSAM's image embedding once per image for a fair comparison.
        predictor.set_image(medsam_image)

        for model_name, prompt_model in model_configs:
            with torch.inference_mode():
                cx, cy, bw, bh = (
                    prompt_model(tensor)[0].cpu().numpy().astype(float)
                )

            pred_box = [
                float(np.clip((cx - bw / 2) * width, 0, width - 1)),
                float(np.clip((cy - bh / 2) * height, 0, height - 1)),
                float(np.clip((cx + bw / 2) * width, 1, width)),
                float(np.clip((cy + bh / 2) * height, 1, height)),
            ]
            pred_box[2] = max(pred_box[2], min(width, pred_box[0] + 1))
            pred_box[3] = max(pred_box[3], min(height, pred_box[1] + 1))

            medsam_box = np.array([
                pred_box[0] * MEDSAM_SIZE / width,
                pred_box[1] * MEDSAM_SIZE / height,
                pred_box[2] * MEDSAM_SIZE / width,
                pred_box[3] * MEDSAM_SIZE / height,
            ], dtype=np.float32)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=medsam_box,
                    multimask_output=False,
                )

            pred_mask = masks[0].astype(bool)
            if pred_mask.shape != (height, width):
                pred_mask = cv2.resize(
                    pred_mask.astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            metrics = calculate_metrics(pred_mask, gt_mask)
            box_iou = bbox_iou_xyxy(pred_box, gt_box)

            record = {
                "model": model_name,
                "filename": image_path.name,
                "pred_x1": pred_box[0],
                "pred_y1": pred_box[1],
                "pred_x2": pred_box[2],
                "pred_y2": pred_box[3],
                "category": str(row.get("category", "")),
                "mask_count": mask_count,
                "lesion_size": lesion_size,
                "gt_mask_area_fraction": area_fraction,
                "medsam_score": float(scores[0]),
                "bbox_iou_vs_mask_bbox": box_iou,
                **metrics,
            }
            results.append(record)

            # Save a few examples per model with distinct filenames.
            if index <= 10:
                safe_stem = image_path.stem.replace(" ", "_")
                save_visual(
                    image_rgb, gt_mask, pred_mask, pred_box,
                    VISUAL_DIR / f"{safe_stem}_{model_name}.png"
                )

        if index % 10 == 0 or index == len(test_df):
            print(
                f"[{index}/{len(test_df)}] {image_path.name} | "
                "evaluated V2 and V3"
            )

    if not results:
        raise RuntimeError("No images were successfully evaluated.")

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)

    summary = result_df.groupby("model").agg(
        images=("filename", "nunique"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        mean_segmentation_iou=("segmentation_iou", "mean"),
        median_segmentation_iou=("segmentation_iou", "median"),
        mean_bbox_iou=("bbox_iou_vs_mask_bbox", "mean"),
    )
    print("\nOVERALL V2 vs V3")
    print(summary.round(4).to_string())

    size_summary = result_df.groupby(
        ["model", "lesion_size"], observed=True
    ).agg(
        images=("filename", "nunique"),
        mean_dice=("dice", "mean"),
        mean_segmentation_iou=("segmentation_iou", "mean"),
        mean_bbox_iou=("bbox_iou_vs_mask_bbox", "mean"),
    )
    print("\nRESULTS BY GROUND-TRUTH LESION SIZE")
    print(size_summary.round(4).to_string())

    summary.to_csv(OUTPUT_DIR / "overall_summary.csv")
    size_summary.to_csv(OUTPUT_DIR / "lesion_size_summary.csv")

    print(f"\nPer-image CSV: {OUTPUT_CSV}")
    print(f"Overall summary: {OUTPUT_DIR / 'overall_summary.csv'}")
    print(f"Lesion-size summary: {OUTPUT_DIR / 'lesion_size_summary.csv'}")
    print(f"Elapsed time: {(time.time() - started) / 60:.1f} minutes")
    print("Research evaluation only; not clinical validation.")


    print("\n" + "=" * 58)
    print("SEGMENTATION EVALUATION SUMMARY")
    print("=" * 58)
    print(f"Successfully evaluated: {len(result_df)}")
    print(f"Mean Dice:              {result_df['dice'].mean():.4f}")
    print(f"Median Dice:            {result_df['dice'].median():.4f}")
    print(f"Mean segmentation IoU:  {result_df['segmentation_iou'].mean():.4f}")
    print(f"Median segmentation IoU:{result_df['segmentation_iou'].median():.4f}")
    print(f"Mean precision:         {result_df['precision'].mean():.4f}")
    print(f"Mean recall:            {result_df['recall'].mean():.4f}")
    print(f"Mean box IoU vs GT mask: {result_df['bbox_iou_vs_mask_bbox'].mean():.4f}")
    print(f"Mean MedSAM score:      {result_df['medsam_score'].mean():.4f}")
    print(f"Elapsed time:           {(time.time() - started) / 60:.1f} minutes")
    print(f"\nPer-image CSV: {OUTPUT_CSV}")
    print(f"Visual examples: {VISUAL_DIR}")
    print("\nVisuals: predicted mask overlay in red, GT contour in cyan, "
          "predicted box in green.")
    print("Note: this is an experimental research evaluation, not clinical validation.")


if __name__ == "__main__":
    main()



