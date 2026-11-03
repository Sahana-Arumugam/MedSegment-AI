from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
from pkg_resources import safe_name
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
MANIFEST = BACKEND / "prepared_dataset" / "busi_manifest.csv"
PROMPT_CHECKPOINT = BACKEND / "models" / "auto_prompt_cnn_v2.pth"
MEDSAM_DIR = ROOT / "MedSAM"
MEDSAM_CHECKPOINT = MEDSAM_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
OUT = BACKEND / "failure_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
MEDSAM_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = {
    "malignant (194).png",
    "benign (377).png",
    "benign (429).png",
    "benign (119).png",
    "malignant (81).png",
}

sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(MEDSAM_DIR))

# Import the exact prompt model and mask loader from the existing evaluator.
from evaluate_segmentation import AutoPromptCNNV2, load_ground_truth_mask

from segment_anything import sam_model_registry, SamPredictor

def main():
    df = pd.read_csv(MANIFEST)
    selected = df[df["image_path"].apply(
        lambda p: Path(str(p)).name in TARGETS
    )].copy()

    found = set(selected["image_path"].apply(
        lambda p: Path(str(p)).name
    ))
    missing = TARGETS - found
    if missing:
        print("NOT FOUND IN MANIFEST:", sorted(missing))

    if selected.empty:
        raise RuntimeError("None of the target images were found in the manifest.")

    prompt_model = AutoPromptCNNV2().to(DEVICE)
    ckpt = torch.load(
        PROMPT_CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    prompt_model.load_state_dict(ckpt["model_state_dict"])
    prompt_model.eval()

    sam = sam_model_registry["vit_b"](
        checkpoint=str(MEDSAM_CHECKPOINT)
    )
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)

    metrics_rows = []

    print("Device:", DEVICE)
    print("Output folder:", OUT)

    for _, row in selected.iterrows():
        image_path = Path(str(row["image_path"]))
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print("SKIPPED: cannot read", image_path)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        mask_paths = [
            p.strip() for p in str(row["mask_paths"]).split(";")
            if p.strip() and p.strip().lower() != "nan"
        ]
        gt_mask, mask_count = load_ground_truth_mask(
            mask_paths, (height, width)
        )
        if mask_count == 0 or not gt_mask.any():
            print("SKIPPED: no usable GT mask", image_path.name)
            continue

        prompt_image = cv2.resize(
            image_rgb, (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0

        tensor = torch.from_numpy(
            prompt_image.transpose(2, 0, 1).copy()
        ).unsqueeze(0).to(DEVICE)

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

        medsam_image = cv2.resize(
            image_rgb, (MEDSAM_SIZE, MEDSAM_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        medsam_box = np.array([
            pred_box[0] * MEDSAM_SIZE / width,
            pred_box[1] * MEDSAM_SIZE / height,
            pred_box[2] * MEDSAM_SIZE / width,
            pred_box[3] * MEDSAM_SIZE / height,
        ], dtype=np.float32)
        # Diagnostic oracle box derived from the ground-truth mask.
        gt_ys, gt_xs = np.where(gt_mask)
        oracle_box = np.array([
            gt_xs.min() * MEDSAM_SIZE / width,
            gt_ys.min() * MEDSAM_SIZE / height,
            (gt_xs.max() + 1) * MEDSAM_SIZE / width,
            (gt_ys.max() + 1) * MEDSAM_SIZE / height,
        ], dtype=np.float32)
        predictor.set_image(medsam_image)
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=medsam_box,
                multimask_output=False,
            )
                # Oracle diagnostic: same image and model, but correct box.
        with torch.inference_mode():
            oracle_masks, oracle_scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=oracle_box,
                multimask_output=False,
            )

        oracle_mask = oracle_masks[0].astype(bool)
        if oracle_mask.shape != (height, width):
            oracle_mask = cv2.resize(
                oracle_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        oracle_tp = int(np.logical_and(oracle_mask, gt_mask).sum())
        oracle_fp = int(np.logical_and(oracle_mask, ~gt_mask).sum())
        oracle_fn = int(np.logical_and(~oracle_mask, gt_mask).sum())

        oracle_dice = (
            2 * oracle_tp / (2 * oracle_tp + oracle_fp + oracle_fn)
            if 2 * oracle_tp + oracle_fp + oracle_fn else 1.0
        )
        oracle_iou = (
            oracle_tp / (oracle_tp + oracle_fp + oracle_fn)
            if oracle_tp + oracle_fp + oracle_fn else 1.0
        )

        print(
            f"  ORACLE BOX: Dice={oracle_dice:.4f}, "
            f"IoU={oracle_iou:.4f}, "
            f"MedSAM score={float(oracle_scores[0]):.4f}"
        )

        pred_mask = masks[0].astype(bool)
        if pred_mask.shape != (height, width):
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        tp = int(np.logical_and(pred_mask, gt_mask).sum())
        fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
        fn = int(np.logical_and(~pred_mask, gt_mask).sum())
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0

        ys, xs = np.where(gt_mask)
        gt_box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        ix1, iy1 = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
        ix2, iy2 = min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_p = max(0, pred_box[2] - pred_box[0]) * max(0, pred_box[3] - pred_box[1])
        area_g = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        box_iou = inter / (area_p + area_g - inter) if area_p + area_g > inter else 0.0

        # Pixels: green = correct prediction, red = false positive,
        # blue = false negative.
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[np.logical_and(pred_mask, gt_mask)] = [0, 220, 0]
        overlay[np.logical_and(pred_mask, ~gt_mask)] = [255, 40, 40]
        overlay[np.logical_and(~pred_mask, gt_mask)] = [40, 100, 255]

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original image")

        axes[1].imshow(image_rgb)
        axes[1].imshow(gt_mask, cmap=ListedColormap(
            [(0, 0, 0, 0), (0, 0.8, 1, 0.55)]
        ), vmin=0, vmax=1)
        axes[1].set_title("Ground-truth mask")

        axes[2].imshow(image_rgb)
        axes[2].imshow(pred_mask, cmap=ListedColormap(
            [(0, 0, 0, 0), (1, 0, 0, 0.5)]
        ), vmin=0, vmax=1)
        axes[2].add_patch(Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            fill=False, edgecolor="lime", linewidth=2
        ))
        axes[2].set_title("Prediction + predicted box")

        axes[3].imshow(overlay)
        axes[3].set_title("Green=TP, red=FP, blue=FN")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{image_path.name} | Dice={dice:.4f} | IoU={iou:.4f} | "
            f"Box IoU={box_iou:.4f} | Score={float(scores[0]):.4f}",
            fontsize=11
        )
        fig.tight_layout()
        safe_name = image_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        output_path = OUT / f"{safe_name}_diagnostic.png"
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        metrics_rows.append({
            "filename": image_path.name,
            "mask_count": mask_count,
            "pred_box_x1": pred_box[0],
            "pred_box_y1": pred_box[1],
            "pred_box_x2": pred_box[2],
            "pred_box_y2": pred_box[3],
            "box_iou": box_iou,
            "medsam_score": float(scores[0]),
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn,
            "diagnostic_image": str(output_path),
        })
        print(
            f"{image_path.name}: Dice={dice:.4f}, IoU={iou:.4f}, "
            f"box IoU={box_iou:.4f}"
        )
        print("Saved:", output_path)

    if metrics_rows:
        csv_path = OUT / "failure_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
        print("Metrics saved:", csv_path)

if __name__ == "__main__":
    main()
