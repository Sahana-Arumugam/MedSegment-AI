from pathlib import Path
import csv
import hashlib
import json
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
BACKEND = PROJECT / "backend"

DATASET = Path(
    r"C:\Users\Sahana Arumugam\Downloads\BUS-UCLM\BUS-UCLM Breast ultrasound lesion segmentation dataset\BUS-UCLM"
)
MANIFEST = DATASET.parent / "integrity_audit" / "frozen_external_candidate_manifest.csv"

PROMPT_CHECKPOINT = BACKEND / "models" / "auto_prompt_cnn_v2.pth"
MEDSAM_DIR = PROJECT / "MedSAM"
MEDSAM_CHECKPOINT = MEDSAM_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"

OUTPUT = BACKEND / "external_evaluation_v1"
IMAGE_SIZE = 256
BOOTSTRAP_REPS = 2000
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def decode_mask(mask_path, expected_shape):
    """Decode BUS-UCLM red/green lesion pixels; black is background."""
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    if mask_bgr.shape[:2] != expected_shape:
        raise ValueError(
            f"Image/mask size mismatch for {mask_path.name}: "
            f"image={expected_shape}, mask={mask_bgr.shape[:2]}"
        )

    # OpenCV loads BGR. Dataset foreground colors are pure red and green.
    blue, green, red = cv2.split(mask_bgr)
    foreground = (red > 0) | (green > 0)

    unexpected = (blue > 0) | (
        (red > 0) & (green > 0)
    )
    if np.any(unexpected):
        raise ValueError(f"Unexpected mask colors found in {mask_path.name}")

    return foreground.astype(bool)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3,
                stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, 3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
                ),
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


def calculate_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    pred_count = int(pred.sum())
    gt_count = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    dice = 2 * tp / (pred_count + gt_count) if pred_count + gt_count else 1.0
    iou = tp / union if union else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0

    return {
        "dice": float(dice),
        "segmentation_iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "predicted_foreground_pixels": pred_count,
        "ground_truth_foreground_pixels": gt_count,
        "predicted_foreground_fraction": float(pred.mean()),
    }


def patient_bootstrap_ci(df, metric, reps=BOOTSTRAP_REPS):
    """Percentile CI by resampling patient-level means, not images."""
    patient_means = (
        df.groupby("patient_id")[metric].mean().dropna().to_numpy()
    )
    if len(patient_means) < 2:
        return {
            "patients": int(len(patient_means)),
            "mean_of_patient_means": (
                float(patient_means.mean()) if len(patient_means) else None
            ),
            "ci95_low": None,
            "ci95_high": None,
        }

    rng = np.random.default_rng(SEED)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(patient_means, size=len(patient_means), replace=True)
        boot[i] = sample.mean()

    return {
        "patients": int(len(patient_means)),
        "mean_of_patient_means": float(patient_means.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def main():
    for path in (MANIFEST, PROMPT_CHECKPOINT, MEDSAM_CHECKPOINT):
        require_file(path)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output folder already exists; refusing to overwrite: {OUTPUT}"
        )

    df = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)

    required = {
        "filename", "patient_id", "image_path", "mask_path",
        "image_sha256", "mask_sha256", "candidate_eligible",
        "excluded_patient_group", "dimensions_match", "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    # Use only frozen eligible rows; do not create a new split or tune on data.
    selected = df[
        df["candidate_eligible"].map(truth)
        & ~df["excluded_patient_group"].map(truth)
        & df["dimensions_match"].map(truth)
    ].copy()

    if selected.empty:
        raise RuntimeError("No eligible records in frozen manifest.")

    if selected["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in candidate manifest.")

    print("Device:", DEVICE)
    print("Frozen manifest rows:", len(df))
    print("Eligible candidate rows:", len(selected))
    print("Patient groups:", selected["patient_id"].nunique())
    print("Class counts:", selected["label"].value_counts().to_dict())
    print("Output:", OUTPUT)

    # Load the already-trained prompt model; never train or save weights here.
    prompt_checkpoint = torch.load(
        PROMPT_CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    if prompt_checkpoint.get("architecture") != "AutoPromptCNNV2":
        raise RuntimeError("Unexpected prompt checkpoint architecture.")
    if int(prompt_checkpoint.get("image_size", -1)) != IMAGE_SIZE:
        raise RuntimeError("Checkpoint image size differs from frozen setting.")

    prompt_model = AutoPromptCNNV2().to(DEVICE)
    prompt_model.load_state_dict(prompt_checkpoint["model_state_dict"])
    prompt_model.eval()

    if str(MEDSAM_DIR) not in sys.path:
        sys.path.insert(0, str(MEDSAM_DIR))
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_b"](checkpoint=str(MEDSAM_CHECKPOINT))
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)

    results = []
    start = time.time()

    for index, (_, row) in enumerate(selected.iterrows(), start=1):
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        require_file(image_path)
        require_file(mask_path)

        # Ensure files still match the frozen manifest.
        if sha256_file(image_path) != row["image_sha256"]:
            raise RuntimeError(f"Image hash mismatch: {image_path.name}")
        if sha256_file(mask_path) != row["mask_sha256"]:
            raise RuntimeError(f"Mask hash mismatch: {mask_path.name}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        gt = decode_mask(mask_path, (height, width))
        is_normal = row["label"].strip().lower() == "normal"

        # Prompt CNN preprocessing exactly matches its training path:
        # resize directly to 256x256, RGB, float in [0,1].
        prompt_image = cv2.resize(
            image_rgb, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0

        tensor = torch.from_numpy(
            prompt_image.transpose(2, 0, 1).copy()
        ).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            cx, cy, bw, bh = (
                prompt_model(tensor)[0].cpu().numpy().astype(float)
            )

        # Original-image coordinates. SamPredictor transforms this box
        # internally to match its aspect-ratio-preserving image resize.
        x1 = float(np.clip((cx - bw / 2) * width, 0, width - 1))
        y1 = float(np.clip((cy - bh / 2) * height, 0, height - 1))
        x2 = float(np.clip((cx + bw / 2) * width, 1, width))
        y2 = float(np.clip((cy + bh / 2) * height, 1, height))
        x2 = max(x2, min(width, x1 + 1))
        y2 = max(y2, min(height, y1 + 1))
        pred_box = np.array([x1, y1, x2, y2], dtype=np.float32)

        # Pass the original uint8 RGB image; predictor handles resizing,
        # padding, and model normalization.
        predictor.set_image(image_rgb)
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=pred_box,
                multimask_output=False,
            )

        pred = masks[0].astype(bool)
        if pred.shape != gt.shape:
            raise RuntimeError(
                f"Prediction size mismatch for {image_path.name}: "
                f"{pred.shape} vs {gt.shape}"
            )

        metrics = calculate_metrics(pred, gt)
        record = {
            "filename": image_path.name,
            "patient_id": row["patient_id"],
            "label": row["label"].strip().lower(),
            "is_normal": is_normal,
            "width": width,
            "height": height,
            "medsam_score": float(scores[0]),
            "pred_x1": x1,
            "pred_y1": y1,
            "pred_x2": x2,
            "pred_y2": y2,
            **metrics,
        }
        results.append(record)

        if index % 10 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {image_path.name} | "
                f"label={record['label']} | "
                f"Dice={metrics['dice']:.4f} | "
                f"predicted pixels={metrics['predicted_foreground_pixels']}"
            )

    result_df = pd.DataFrame(results)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result_df.to_csv(OUTPUT / "per_image_metrics.csv", index=False)

    lesion_df = result_df[~result_df["is_normal"]].copy()
    normal_df = result_df[result_df["is_normal"]].copy()

    summary = {
        "protocol": "Frozen zero-shot external baseline; no BUS-UCLM training or tuning",
        "manifest_sha256": sha256_file(MANIFEST),
        "prompt_checkpoint_sha256": sha256_file(PROMPT_CHECKPOINT),
        "medsam_checkpoint_sha256": sha256_file(MEDSAM_CHECKPOINT),
        "device": str(DEVICE),
        "image_size_prompt_model": IMAGE_SIZE,
        "medsam_image_input": "original uint8 RGB image; SamPredictor handles resize",
        "bbox_coordinates": "original-image XYXY pixels",
        "eligible_images": int(len(result_df)),
        "patient_groups": int(result_df["patient_id"].nunique()),
        "class_counts": {
            str(k): int(v) for k, v in result_df["label"].value_counts().items()
        },
        "lesion_cases": int(len(lesion_df)),
        "normal_cases": int(len(normal_df)),
        "lesion_image_level": {},
        "lesion_patient_bootstrap": {},
        "normal_image_level": {},
        "by_label": {},
        "elapsed_minutes": float((time.time() - start) / 60),
    }

    if len(lesion_df):
        for metric in ("dice", "segmentation_iou", "precision", "recall"):
            summary["lesion_image_level"][metric + "_mean"] = float(
                lesion_df[metric].mean()
            )
            summary["lesion_image_level"][metric + "_median"] = float(
                lesion_df[metric].median()
            )
            summary["lesion_patient_bootstrap"][metric] = patient_bootstrap_ci(
                lesion_df, metric
            )

    if len(normal_df):
        summary["normal_image_level"] = {
            "false_positive_image_rate": float(
                (normal_df["predicted_foreground_pixels"] > 0).mean()
            ),
            "mean_predicted_foreground_fraction": float(
                normal_df["predicted_foreground_fraction"].mean()
            ),
            "median_predicted_foreground_fraction": float(
                normal_df["predicted_foreground_fraction"].median()
            ),
            "mean_false_positive_pixels": float(
                normal_df["fp_pixels"].mean()
            ),
        }

    for label, group in result_df.groupby("label"):
        summary["by_label"][str(label)] = {
            "images": int(len(group)),
            "patients": int(group["patient_id"].nunique()),
            "mean_dice": float(group["dice"].mean()),
            "median_dice": float(group["dice"].median()),
            "false_positive_image_rate": float(
                (group["predicted_foreground_pixels"] > 0).mean()
            ) if (group["is_normal"]).any() else None,
        }

    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===== EXTERNAL BASELINE SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nPer-image results:", OUTPUT / "per_image_metrics.csv")
    print("Summary:", OUTPUT / "summary.json")
    print("No model weights were modified or saved.")


if __name__ == "__main__":
    main()

