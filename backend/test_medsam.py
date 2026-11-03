
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDSAM_ROOT = PROJECT_ROOT / "MedSAM"
CHECKPOINT = MEDSAM_ROOT / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "test_outputs"

IMAGE_PATH = PROJECT_ROOT / "data" / "BUSI" / "benign" / "benign (1).png"
GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "BUSI" / "benign" / "benign (1)_mask.png"
)

sys.path.insert(0, str(MEDSAM_ROOT))
from segment_anything import sam_model_registry


def calculate_metrics(prediction, ground_truth):
    pred = prediction > 0
    gt = ground_truth > 0

    intersection = np.logical_and(pred, gt).sum()
    pred_count = pred.sum()
    gt_count = gt.sum()
    union = np.logical_or(pred, gt).sum()

    dice = (
        2.0 * intersection / (pred_count + gt_count)
        if pred_count + gt_count
        else 1.0
    )
    iou = intersection / union if union else 1.0

    return float(dice), float(iou)


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {GROUND_TRUTH_PATH}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load image and annotation
    image_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(str(GROUND_TRUTH_PATH), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or mask_gray is None:
        raise RuntimeError("Could not read the image or ground-truth mask.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    if mask_gray.shape != (height, width):
        raise ValueError(
            f"Image size {(height, width)} does not match mask size {mask_gray.shape}"
        )

    ground_truth = (mask_gray > 0).astype(np.uint8)

    # Find a bounding box around all white pixels in the annotation.
    ys, xs = np.where(ground_truth > 0)
    if len(xs) == 0:
        raise ValueError("The ground-truth mask contains no white pixels.")

    # Use exclusive upper coordinates for the box.
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

    print(f"Image: {IMAGE_PATH.name}")
    print(f"Dimensions: {width} x {height}")
    print(f"Ground-truth box: [{x1}, {y1}, {x2}, {y2}]")
    print("Loading MedSAM on CPU...")

    model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
    model = model.to("cpu")
    model.eval()

    # Match the official MedSAM preprocessing.
    resized = cv2.resize(
        image_rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)

    value_range = float(resized.max() - resized.min())
    resized = (resized - resized.min()) / max(value_range, 1e-8)

    image_tensor = (
        torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    )

    # Scale the annotation-derived box to the 1024 x 1024 model coordinates.
    box_1024 = np.array(
        [
            x1 * 1024.0 / width,
            y1 * 1024.0 / height,
            x2 * 1024.0 / width,
            y2 * 1024.0 / height,
        ],
        dtype=np.float32,
    )
    box_tensor = torch.from_numpy(box_1024).reshape(1, 1, 4)

    print("Running oracle-box inference...")

    with torch.inference_mode():
        image_embedding = model.image_encoder(image_tensor)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None,
        )
        low_res_logits, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        probability = torch.nn.functional.interpolate(
            torch.sigmoid(low_res_logits),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

    prediction = (probability > 0.5).astype(np.uint8)
    dice, iou = calculate_metrics(prediction, ground_truth)

    # Save binary masks.
    pred_path = OUTPUT_DIR / "oracle_box_prediction.png"
    gt_path = OUTPUT_DIR / "ground_truth.png"
    overlay_path = OUTPUT_DIR / "oracle_box_overlay.png"
    comparison_path = OUTPUT_DIR / "oracle_box_comparison.png"

    cv2.imwrite(str(pred_path), prediction * 255)
    cv2.imwrite(str(gt_path), ground_truth * 255)

    # Red = predicted mask; green = ground truth; yellow = overlap.
    comparison = np.zeros((height, width, 3), dtype=np.uint8)
    comparison[:, :, 2] = prediction * 255
    comparison[:, :, 1] = ground_truth * 255
    cv2.imwrite(str(comparison_path), comparison)

    overlay = image_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = prediction * 255
    overlay = cv2.addWeighted(overlay, 1.0, red, 0.35, 0)
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 2)
    cv2.imwrite(str(overlay_path), overlay)

    print("\nOracle-box baseline completed.")
    print(f"Dice: {dice:.4f}")
    print(f"IoU:  {iou:.4f}")
    print(f"Predicted foreground pixels: {int(prediction.sum())}")
    print(f"Ground-truth foreground pixels: {int(ground_truth.sum())}")
    print(f"Prediction: {pred_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Comparison: {comparison_path}")
    print(
        "\nThese metrics are for one image and an annotation-derived box. "
        "They are not test-set results or evidence of clinical performance."
    )


if __name__ == "__main__":
    main()
