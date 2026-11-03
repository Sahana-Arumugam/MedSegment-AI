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
OUT_CSV = OUT_DIR / "geometry_analysis.csv"

IMAGE_SIZE = 256
MEDSAM_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(MEDSAM_DIR))

from evaluate_segmentation import AutoPromptCNNV2, load_ground_truth_mask
from segment_anything import sam_model_registry, SamPredictor


def dice_iou(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    denom = 2 * tp + fp + fn
    union = tp + fp + fn
    return (
        float(2 * tp / denom) if denom else 1.0,
        float(tp / union) if union else 1.0,
    )


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


def geometry(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = box

    area = int(mask.sum())
    image_area = max(h * w, 1)

    bx1 = int(np.clip(np.floor(x1), 0, w - 1))
    by1 = int(np.clip(np.floor(y1), 0, h - 1))
    bx2 = int(np.clip(np.ceil(x2), bx1 + 1, w))
    by2 = int(np.clip(np.ceil(y2), by1 + 1, h))

    box_area = max((bx2 - bx1) * (by2 - by1), 1)
    inside = np.zeros_like(mask, dtype=bool)
    inside[by1:by2, bx1:bx2] = True

    labels_count, labels = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8
    )
    component_areas = np.bincount(labels.ravel())[1:]
    nonzero_components = int(np.count_nonzero(component_areas))

    # Fraction of mask pixels that fall outside the prompting box.
    outside_pixels = int(np.logical_and(mask, ~inside).sum())

    # Boundary contact is measured along the edges of the clipped box.
    edge = np.zeros_like(mask, dtype=bool)
    edge[by1, bx1:bx2] = True
    edge[by2 - 1, bx1:bx2] = True
    edge[by1:by2, bx1] = True
    edge[by1:by2, bx2 - 1] = True
    mask_touches_box_edge = bool(np.logical_and(mask, edge).any())

    return {
        "mask_area_pixels": area,
        "mask_area_image_fraction": area / image_area,
        "mask_area_box_fraction": area / box_area,
        "connected_components": nonzero_components,
        "largest_component_fraction": (
            float(component_areas.max() / area) if area else 0.0
        ),
        "mask_fraction_outside_box": (
            outside_pixels / area if area else 0.0
        ),
        "mask_touches_box_boundary": mask_touches_box_edge,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)
    val = df[
        (df["split"].astype(str).str.lower() == "validation")
        & (df["has_mask"].astype(str).str.lower() == "true")
        & (df["has_bbox"].astype(str).str.lower() == "true")
    ].copy()

    print("Device:", DEVICE)
    print("Eligible validation images:", len(val))
    print("Output:", OUT_CSV)

    prompt = AutoPromptCNNV2().to(DEVICE)
    ckpt = torch.load(
        PROMPT_CKPT, map_location=DEVICE, weights_only=False
    )
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
            item.strip()
            for item in str(row["mask_paths"]).split(";")
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

        candidates = {
            "base": box,
            "expanded": adjust_box(box, 1.10, w, h),
            "contracted": adjust_box(box, 0.90, w, h),
        }

        record = {"filename": path.name}

        for name, candidate_box in candidates.items():
            scaled = candidate_box * np.array([
                MEDSAM_SIZE / w, MEDSAM_SIZE / h,
                MEDSAM_SIZE / w, MEDSAM_SIZE / h,
            ], dtype=np.float32)

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=scaled,
                    multimask_output=False,
                )

            mask = masks[0].astype(bool)
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            dice, iou = dice_iou(mask, gt)
            features = geometry(mask, candidate_box)

            record[f"{name}_dice"] = dice
            record[f"{name}_iou"] = iou
            record[f"{name}_medsam_score"] = float(scores[0])

            for feature_name, value in features.items():
                record[f"{name}_{feature_name}"] = value

        rows.append(record)

        if idx % 10 == 0 or idx == len(val):
            print(f"[{idx}/{len(val)}] {path.name}")

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print("\nSaved:", OUT_CSV)
    print("Successfully evaluated:", len(result))
    print("Columns:", len(result.columns))


if __name__ == "__main__":
    main()
