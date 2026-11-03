from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
MANIFEST = BACKEND / "prepared_dataset" / "busi_manifest.csv"
PROMPT_CKPT = BACKEND / "models" / "auto_prompt_cnn_v2.pth"
MEDSAM_DIR = ROOT / "MedSAM"
MEDSAM_CKPT = MEDSAM_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
OUT_DIR = BACKEND / "prompt_stability_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
MEDSAM_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(MEDSAM_DIR))

from evaluate_segmentation import AutoPromptCNNV2, load_ground_truth_mask
from segment_anything import sam_model_registry, SamPredictor


def dice_iou(pred, gt):
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return dice, iou


def adjust_box(box, factor, width, height):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    return np.array([
        np.clip(cx - bw / 2, 0, width - 1),
        np.clip(cy - bh / 2, 0, height - 1),
        np.clip(cx + bw / 2, 1, width),
        np.clip(cy + bh / 2, 1, height),
    ], dtype=np.float32)


def mask_stability(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def main():
    df = pd.read_csv(MANIFEST)
    val = df[
        (df["split"].astype(str).str.lower() == "validation")
        & (df["has_mask"].astype(str).str.lower() == "true")
        & (df["has_bbox"].astype(str).str.lower() == "true")
    ].copy()

    print("Device:", DEVICE)
    print("Eligible validation images:", len(val))

    prompt = AutoPromptCNNV2().to(DEVICE)
    ckpt = torch.load(PROMPT_CKPT, map_location=DEVICE, weights_only=False)
    prompt.load_state_dict(ckpt["model_state_dict"])
    prompt.eval()

    sam = sam_model_registry["vit_b"](checkpoint=str(MEDSAM_CKPT))
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)

    rows = []
    for idx, (_, row) in enumerate(val.iterrows(), start=1):
        path = Path(str(row["image_path"]))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print("Skipping unreadable:", path.name)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        mask_paths = [
            item.strip() for item in str(row["mask_paths"]).split(";")
            if item.strip() and item.strip().lower() != "nan"
        ]
        gt, count = load_ground_truth_mask(mask_paths, (h, w))
        if count == 0 or not gt.any():
            print("Skipping empty/unusable mask:", path.name)
            continue

        small = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(small.transpose(2, 0, 1).copy())
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            cx, cy, bw, bh = prompt(tensor)[0].cpu().numpy().astype(float)

        box = np.array([
            np.clip((cx - bw / 2) * w, 0, w - 1),
            np.clip((cy - bh / 2) * h, 0, h - 1),
            np.clip((cx + bw / 2) * w, 1, w),
            np.clip((cy + bh / 2) * h, 1, h),
        ], dtype=np.float32)

        box[2] = max(box[2], min(w, box[0] + 1))
        box[3] = max(box[3], min(h, box[1] + 1))

        med_rgb = cv2.resize(rgb, (MEDSAM_SIZE, MEDSAM_SIZE))
        predictor.set_image(med_rgb)

        def infer(original_box):
            scaled = original_box * np.array([
                MEDSAM_SIZE / w, MEDSAM_SIZE / h,
                MEDSAM_SIZE / w, MEDSAM_SIZE / h
            ], dtype=np.float32)
            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=scaled, multimask_output=False
                )
            mask = masks[0].astype(bool)
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            return mask, float(scores[0])

        base_mask, base_score = infer(box)
        expanded_mask, expanded_score = infer(adjust_box(box, 1.10, w, h))
        contracted_mask, contracted_score = infer(adjust_box(box, 0.90, w, h))

        base_dice, base_iou = dice_iou(base_mask, gt)
        exp_dice, exp_iou = dice_iou(expanded_mask, gt)
        con_dice, con_iou = dice_iou(contracted_mask, gt)

        stability_exp = mask_stability(base_mask, expanded_mask)
        stability_con = mask_stability(base_mask, contracted_mask)

        rows.append({
            "filename": path.name,
            "base_dice": base_dice,
            "base_iou": base_iou,
            "expanded_dice": exp_dice,
            "contracted_dice": con_dice,
            "expanded_mask_iou_vs_base": stability_exp,
            "contracted_mask_iou_vs_base": stability_con,
            "mean_mask_stability": (stability_exp + stability_con) / 2,
            "base_medsam_score": base_score,
            "expanded_medsam_score": expanded_score,
            "contracted_medsam_score": contracted_score,
        })

        if idx % 10 == 0 or idx == len(val):
            print(f"[{idx}/{len(val)}] {path.name} | Dice={base_dice:.4f}")

    out_csv = OUT_DIR / "validation_prompt_stability.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("\nResults saved:", out_csv)
    print("Successfully evaluated:", len(rows))


if __name__ == "__main__":
    main()
