
from pathlib import Path
from urllib.parse import quote
from functools import lru_cache
import base64
import csv
import html
import io
import sys
import time
import json
import uuid
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import FileResponse, StreamingResponse

from backend.experiment_store import (
    create_experiment,
    complete_experiment,
    list_experiments,
    get_experiment,
    get_artifact_path,
    update_experiment_metrics,
)

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as ReportImage, PageBreak, KeepTogether,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
DATASET_DIR = PROJECT_DIR / "data" / "BUSI"

MEDSAM_DIR = PROJECT_DIR / "MedSAM"
MEDSAM_CHECKPOINT = (
    MEDSAM_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
)

# Use the improved v2 prompt generator.
PROMPT_CHECKPOINT = (
    BACKEND_DIR / "models" / "auto_prompt_cnn_v2.pth"
)

IMAGE_SIZE = 256
MEDSAM_SIZE = 1024

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reviewed masks are stored separately from experiment predictions.
REVIEWED_MASKS_DIR = BACKEND_DIR / "reviewed_annotations"
REVIEWED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MedSegment AI API",
    version="0.3.0",
    description=(
        "BUSI dataset explorer and experimental "
        "medical image segmentation inference."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if not DATASET_DIR.is_dir():
    raise RuntimeError(
        f"BUSI dataset folder not found: {DATASET_DIR}"
    )

app.mount(
    "/dataset-files",
    StaticFiles(directory=str(DATASET_DIR)),
    name="dataset-files",
)


# ============================================================
# V2 AUTOMATIC PROMPT GENERATOR
# Architecture must match the v2 training script.
# Output: normalized center_x, center_y, width, height.
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if in_channels == out_channels and stride == 1:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return torch.relu(
            self.block(x) + self.skip(x)
        )


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
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
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


# ============================================================
# DATASET HELPERS
# ============================================================

def file_url(category: str, filename: str) -> str:
    return (
        "/dataset-files/"
        + quote(category, safe="")
        + "/"
        + quote(filename, safe="")
    )


def is_mask_file(filename: str) -> bool:
    stem = Path(filename).stem.lower()

    return "_mask" in stem and (
        stem.endswith("_mask")
        or "_mask_" in stem
    )


def find_busi_ground_truth(filename: str, image_rgb: np.ndarray):
    """
    Match the uploaded image to a BUSI source image by filename and dimensions.
    Returns the first associated mask and its dataset path, or (None, None).
    A matching filename alone is not treated as enough if dimensions differ.
    """
    if not filename:
        return None, None

    matches = []
    for category in ("benign", "malignant", "normal"):
        category_dir = DATASET_DIR / category
        source_path = category_dir / Path(filename).name
        if not source_path.is_file():
            continue
        try:
            dataset_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if dataset_bgr is None:
                continue
            dataset_rgb = cv2.cvtColor(dataset_bgr, cv2.COLOR_BGR2RGB)
            if dataset_rgb.shape[:2] != image_rgb.shape[:2]:
                continue
            masks = sorted(
                path for path in category_dir.glob("*.png")
                if is_mask_file(path.name)
                and (
                    path.stem == source_path.stem + "_mask"
                    or path.stem.startswith(source_path.stem + "_mask_")
                )
            )
            if masks:
                matches.append((masks[0], source_path))
        except Exception:
            continue

    # If same-named images exist in multiple categories, don't guess.
    if len(matches) != 1:
        return None, None

    mask_path, source_path = matches[0]
    try:
        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        # Ensure the upload is the same BUSI image, not an unrelated file
        # that happens to have the same name and dimensions.
        if not np.array_equal(source_rgb, image_rgb):
            return None, None
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            return None, None
        return gt, mask_path
    except Exception:
        return None, None


def calculate_segmentation_metrics(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray):
    """Compute pixel-wise Dice, IoU, precision, and recall for binary masks."""
    prediction = np.asarray(predicted_mask) > 0
    ground_truth = np.asarray(ground_truth_mask) > 0

    if prediction.shape != ground_truth.shape:
        ground_truth = cv2.resize(
            ground_truth.astype(np.uint8),
            (prediction.shape[1], prediction.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    tp = int(np.logical_and(prediction, ground_truth).sum())
    fp = int(np.logical_and(prediction, np.logical_not(ground_truth)).sum())
    fn = int(np.logical_and(np.logical_not(prediction), ground_truth).sum())

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    # For two empty masks, overlap is defined as perfect; precision/recall
    # are defined as 1 only when there are no predicted/true positives.
    dice = (2 * tp / dice_denominator) if dice_denominator else 1.0
    iou = (tp / iou_denominator) if iou_denominator else 1.0
    precision = (tp / precision_denominator) if precision_denominator else (1.0 if recall_denominator == 0 else 0.0)
    recall = (tp / recall_denominator) if recall_denominator else (1.0 if precision_denominator == 0 else 0.0)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
    }


# ============================================================
# LOAD PROMPT GENERATOR AND MEDSAM
# ============================================================

@lru_cache(maxsize=1)
def load_models():
    if not PROMPT_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"V2 prompt model checkpoint not found: "
            f"{PROMPT_CHECKPOINT}"
        )

    if not MEDSAM_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"MedSAM checkpoint not found: "
            f"{MEDSAM_CHECKPOINT}"
        )

    # Import the official MedSAM implementation.
    if str(MEDSAM_DIR) not in sys.path:
        sys.path.insert(0, str(MEDSAM_DIR))

    from segment_anything import (
        sam_model_registry,
        SamPredictor,
    )

    # Load the v2 automatic prompt generator.
    prompt_model = AutoPromptCNNV2().to(DEVICE)

    saved = torch.load(
        PROMPT_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" not in saved:
        raise RuntimeError(
            "The v2 checkpoint does not contain "
            "'model_state_dict'. Check the training script "
            "and checkpoint format."
        )

    prompt_model.load_state_dict(
        saved["model_state_dict"]
    )
    prompt_model.eval()

    # Load MedSAM ViT-B.
    sam = sam_model_registry["vit_b"](
        checkpoint=str(MEDSAM_CHECKPOINT)
    )

    sam.to(device=DEVICE)
    sam.eval()

    predictor = SamPredictor(sam)

    return prompt_model, predictor


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "MedSegment AI API is running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_found": DATASET_DIR.is_dir(),
        "dataset_path": str(DATASET_DIR),
        "device": str(DEVICE),
        "prompt_model_available": PROMPT_CHECKPOINT.is_file(),
        "prompt_model_version": "AutoPromptCNNV2",
        "prompt_checkpoint": str(PROMPT_CHECKPOINT),
        "medsam_checkpoint_available": MEDSAM_CHECKPOINT.is_file(),
    }


# ============================================================
# EXPERIMENT HISTORY ENDPOINTS
# ============================================================

@app.get("/api/experiments")
def api_list_experiments(limit: int = 100):
    limit = max(1, min(limit, 500))
    return {"experiments": list_experiments(limit)}


@app.get("/api/experiments/{experiment_id}")
def api_get_experiment(experiment_id: str):
    experiment = get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return experiment


@app.get("/api/experiments/{experiment_id}/input")
def api_get_experiment_input(experiment_id: str):
    path = get_artifact_path(experiment_id, "input")
    if path is None:
        raise HTTPException(status_code=404, detail="Input image not found.")
    return FileResponse(path)


@app.get("/api/experiments/{experiment_id}/prediction")
def api_get_experiment_prediction(experiment_id: str):
    path = get_artifact_path(experiment_id, "prediction")
    if path is None:
        raise HTTPException(status_code=404, detail="Predicted mask not found.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/experiments/{experiment_id}/evaluate")
def api_evaluate_experiment(experiment_id: str):
    """Recalculate metrics for a saved run if its input matches a BUSI image."""
    experiment = get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    input_path = get_artifact_path(experiment_id, "input")
    prediction_path = get_artifact_path(experiment_id, "prediction")
    if input_path is None or prediction_path is None:
        raise HTTPException(status_code=404, detail="Saved input or prediction artifact not found.")

    try:
        input_bytes = np.fromfile(str(input_path), dtype=np.uint8)
        prediction_bytes = np.fromfile(str(prediction_path), dtype=np.uint8)
        input_bgr = cv2.imdecode(input_bytes, cv2.IMREAD_COLOR)
        prediction = cv2.imdecode(prediction_bytes, cv2.IMREAD_GRAYSCALE)
        if input_bgr is None or prediction is None:
            raise ValueError("Could not decode saved input or prediction.")
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        ground_truth, ground_truth_path = find_busi_ground_truth(
            experiment["filename"], input_rgb
        )
        if ground_truth is None:
            raise HTTPException(
                status_code=409,
                detail="No unambiguous exact BUSI source-image and ground-truth mask match was found. Metrics were not calculated."
            )
        metrics = calculate_segmentation_metrics(prediction, ground_truth)
        update_experiment_metrics(experiment_id, metrics)
        return {
            "experiment_id": experiment_id,
            "ground_truth_file": ground_truth_path.name,
            **metrics,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metric evaluation failed: {exc}") from exc


# ============================================================
# DATASET ENDPOINT
# ============================================================

@app.get("/api/dataset")
def get_dataset():
    categories = ["benign", "malignant", "normal"]

    images = []
    category_counts = {}

    for category in categories:
        category_dir = DATASET_DIR / category

        if not category_dir.is_dir():
            category_counts[category] = 0
            continue

        files = sorted(
            path
            for path in category_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".png"
        )

        masks = [
            path for path in files
            if is_mask_file(path.name)
        ]

        source_images = [
            path for path in files
            if not is_mask_file(path.name)
        ]

        for image_path in source_images:
            stem = image_path.stem

            matching_masks = [
                mask
                for mask in masks
                if (
                    mask.stem == stem + "_mask"
                    or mask.stem.startswith(stem + "_mask_")
                )
            ]

            images.append({
                "id": f"{category}/{image_path.name}",
                "filename": image_path.name,
                "category": category,
                "image_url": file_url(
                    category, image_path.name
                ),
                "mask_urls": [
                    file_url(category, mask.name)
                    for mask in matching_masks
                ],
                "has_mask": bool(matching_masks),
            })

        category_counts[category] = len(source_images)

    return {
        "total_images": len(images),
        "categories": category_counts,
        "images": images,
    }


# ============================================================
# IMAGE METADATA ENDPOINT
# ============================================================

@app.get("/api/dataset/{category}/{filename}")
def get_image_metadata(category: str, filename: str):
    if category not in {"benign", "malignant", "normal"}:
        raise HTTPException(
            status_code=404,
            detail="Unknown category",
        )

    target = (
        DATASET_DIR / category / filename
    ).resolve()

    category_root = (
        DATASET_DIR / category
    ).resolve()

    if target.parent != category_root or not target.is_file():
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )

    return {
        "filename": target.name,
        "category": category,
        "is_mask": is_mask_file(target.name),
    }


# ============================================================
# SEGMENTATION ENDPOINT
# ============================================================

@app.post("/api/segment")
async def segment_image(file: UploadFile = File(...)):
    run_started = time.perf_counter()
    allowed = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPEG, BMP, or TIFF image.",
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 15 MB or smaller.",
        )

    image_array = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image_array is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image.",
        )

    original_rgb = cv2.cvtColor(
        image_array,
        cv2.COLOR_BGR2RGB,
    )

    original_h, original_w = original_rgb.shape[:2]

    # Load models lazily on the first segmentation request.
    try:
        prompt_model, predictor = load_models()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Models could not be loaded: {exc}",
        ) from exc

    # --------------------------------------------------------
    # Step 1: Predict bounding box with AutoPromptCNNV2.
    # Match v2 training preprocessing: RGB, 256x256, [0, 1].
    # --------------------------------------------------------

    prompt_image = cv2.resize(
        original_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0

    prompt_tensor = torch.from_numpy(
        prompt_image.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        box_prediction = (
            prompt_model(prompt_tensor)[0]
            .cpu()
            .numpy()
        )

    # Model output is normalized center_x, center_y, width, height.
    cx, cy, bw, bh = [
        float(value) for value in box_prediction
    ]

    # Convert normalized box coordinates to original image pixels.
    x1 = (cx - bw / 2.0) * original_w
    y1 = (cy - bh / 2.0) * original_h
    x2 = (cx + bw / 2.0) * original_w
    y2 = (cy + bh / 2.0) * original_h

    # Clip coordinates to image boundaries.
    x1 = float(np.clip(x1, 0, original_w - 1))
    y1 = float(np.clip(y1, 0, original_h - 1))

    x2 = float(np.clip(x2, x1 + 1, original_w))
    y2 = float(np.clip(y2, y1 + 1, original_h))

    # --------------------------------------------------------
    # Step 2: Resize image and box for MedSAM.
    # SamPredictor handles the model's internal preprocessing.
    # --------------------------------------------------------

    medsam_image = cv2.resize(
        original_rgb,
        (MEDSAM_SIZE, MEDSAM_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    scale_x = MEDSAM_SIZE / original_w
    scale_y = MEDSAM_SIZE / original_h

    medsam_box = np.array(
        [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Step 3: Generate segmentation mask with MedSAM.
    # --------------------------------------------------------

    try:
        predictor.set_image(medsam_image)

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=medsam_box,
                multimask_output=False,
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation inference failed: {exc}",
        ) from exc

    # Convert binary mask to an 8-bit PNG at original resolution.
    mask = masks[0].astype(np.uint8) * 255

    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # --------------------------------------------------------
    # Step 4: Encode mask as base64 PNG for the frontend.
    # --------------------------------------------------------

    buffer = io.BytesIO()

    Image.fromarray(mask).save(
        buffer,
        format="PNG",
    )

    mask_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    inference_seconds = time.perf_counter() - run_started

    # Evaluate only when the uploaded pixels match one BUSI source image exactly.
    # The first matching annotation is used, matching the current UI behavior.
    ground_truth_mask, ground_truth_path = find_busi_ground_truth(
        file.filename or "", original_rgb
    )
    metrics = (
        calculate_segmentation_metrics(mask, ground_truth_mask)
        if ground_truth_mask is not None
        else None
    )

    experiment_id = create_experiment(
        filename=file.filename or "uploaded_image",
        model="MedSAM ViT-B + AutoPromptCNNV2",
        prompt_checkpoint=str(PROMPT_CHECKPOINT),
        medsam_checkpoint=str(MEDSAM_CHECKPOINT),
        device=str(DEVICE),
        config={
            "image_size": IMAGE_SIZE,
            "medsam_size": MEDSAM_SIZE,
            "prompt_model": "AutoPromptCNNV2",
            "segmentation_model": "MedSAM ViT-B",
            "prompt_type": "predicted bounding box",
        },
    )

    complete_experiment(
        experiment_id,
        input_bytes=raw,
        mask_png_bytes=buffer.getvalue(),
        inference_seconds=inference_seconds,
        medsam_score=float(scores[0]),
        bbox=[x1, y1, x2, y2],
        metrics=metrics,
    )

    return {
        "experiment_id": experiment_id,
        "inference_seconds": round(inference_seconds, 4),
        "filename": file.filename,
        "width": original_w,
        "height": original_h,
        "device": str(DEVICE),
        "bbox_xyxy": [
            round(x1, 2),
            round(y1, 2),
            round(x2, 2),
            round(y2, 2),
        ],
        "mask_png_base64": mask_base64,
        "mask_mime_type": "image/png",
        "medsam_score": float(scores[0]),
        "metrics": metrics,
        "ground_truth_mask_file": ground_truth_path.name if ground_truth_path else None,
        "foreground_pixels": int(np.count_nonzero(mask)),
        "model": (
            "MedSAM ViT-B with AutoPromptCNNV2-predicted "
            "bounding box"
        ),
        "note": (
            "Experimental segmentation output, not a medical "
            "diagnosis. Segmentation quality has not been "
            "established for arbitrary uploaded images."
        ),
    }



# ============================================================
# SEGMENTATION RELIABILITY STUDIO
# Separate endpoint: does not modify /api/segment or its artifacts.
# ============================================================

def _reliability_encode_png(array: np.ndarray) -> str:
    """Encode a grayscale image or binary mask as base64 PNG."""
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.post("/api/reliability")
async def segmentation_reliability(file: UploadFile = File(...)):
    """
    Generate multiple MedSAM candidate masks, a disagreement map,
    and a morphologically refined copy of the highest-scoring mask.

    This endpoint does not create or overwrite experiment history artifacts.
    """
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPEG, BMP, or TIFF image.",
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 15 MB or smaller.",
        )

    image_array = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image_array is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    original_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    original_h, original_w = original_rgb.shape[:2]

    try:
        prompt_model, predictor = load_models()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Models could not be loaded: {exc}",
        ) from exc

    # Step 1: predict a normalized center_x, center_y, width, height box.
    prompt_image = cv2.resize(
        original_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0

    prompt_tensor = torch.from_numpy(
        prompt_image.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(DEVICE)

    try:
        with torch.inference_mode():
            box_prediction = (
                prompt_model(prompt_tensor)[0].cpu().numpy()
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Automatic prompt generation failed: {exc}",
        ) from exc

    cx, cy, bw, bh = [float(value) for value in box_prediction]

    x1 = float(np.clip((cx - bw / 2.0) * original_w, 0, original_w - 1))
    y1 = float(np.clip((cy - bh / 2.0) * original_h, 0, original_h - 1))
    x2 = float(np.clip((cx + bw / 2.0) * original_w, x1 + 1, original_w))
    y2 = float(np.clip((cy + bh / 2.0) * original_h, y1 + 1, original_h))

    medsam_image = cv2.resize(
        original_rgb,
        (MEDSAM_SIZE, MEDSAM_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    scale_x = MEDSAM_SIZE / original_w
    scale_y = MEDSAM_SIZE / original_h

    medsam_box = np.array(
        [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
        dtype=np.float32,
    )

    # Step 2: request multiple candidate masks from MedSAM.
    try:
        predictor.set_image(medsam_image)

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=medsam_box,
                multimask_output=True,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Candidate mask generation failed: {exc}",
        ) from exc

    if masks is None or len(masks) == 0:
        raise HTTPException(
            status_code=500,
            detail="MedSAM did not return any candidate masks.",
        )

    # Resize every candidate to the uploaded image's dimensions.
    candidate_binary_masks = []

    for candidate in masks:
        binary = (candidate > 0).astype(np.uint8) * 255
        binary = cv2.resize(
            binary,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )
        candidate_binary_masks.append(binary)

    # Select the highest-scoring candidate as the before-refinement mask.
    best_index = int(np.argmax(scores))
    original_mask = candidate_binary_masks[best_index].copy()

    # Step 3: disagreement estimate.
    # Each pixel indicates the fraction of candidates marking it foreground.
    stacked = np.stack(
        [(mask > 0).astype(np.float32) for mask in candidate_binary_masks],
        axis=0,
    )
    foreground_fraction = stacked.mean(axis=0)

    # 0 = unanimous background/foreground; 255 = maximal disagreement.
    disagreement = (
        4.0 * foreground_fraction * (1.0 - foreground_fraction) * 255.0
    ).clip(0, 255).astype(np.uint8)

    # Step 4: conservative morphological closing on a copy.
    # This may smooth small gaps, but is not guaranteed to improve accuracy.
    binary_for_refinement = (original_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    refined_binary = cv2.morphologyEx(
        binary_for_refinement,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    refined_mask = refined_binary.astype(np.uint8) * 255

    # Encode the images for the React page.
    encoded_candidates = [
        _reliability_encode_png(mask)
        for mask in candidate_binary_masks
    ]

    return {
        "filename": file.filename or "uploaded_image",
        "width": original_w,
        "height": original_h,
        "bbox_xyxy": [
            round(x1, 2),
            round(y1, 2),
            round(x2, 2),
            round(y2, 2),
        ],
        "candidate_masks": encoded_candidates,
        "candidate_scores": [
            float(score) for score in scores
        ],
        "uncertainty_map": _reliability_encode_png(disagreement),
        "original_mask": _reliability_encode_png(original_mask),
        "refined_mask": _reliability_encode_png(refined_mask),
        "original_foreground_pixels": int(np.count_nonzero(original_mask)),
        "refined_foreground_pixels": int(np.count_nonzero(refined_mask)),
        "refinement": (
            "Applied one 3x3 elliptical morphological closing operation "
            "to a copy of the highest-scoring candidate mask."
        ),
        "note": (
            "Candidate disagreement is a heuristic visualization, not "
            "calibrated confidence. Refinement is experimental and may "
            "reduce segmentation accuracy. Validate against ground truth."
        ),
    }


# ============================================================
# INTERACTIVE MASK CORRECTION: REVIEWED ANNOTATIONS
# Original experiment prediction files are never overwritten.
# ============================================================

def _review_metadata_path(review_id: str) -> Path:
    return REVIEWED_MASKS_DIR / f"{review_id}.json"


def _review_mask_path(review_id: str) -> Path:
    return REVIEWED_MASKS_DIR / f"{review_id}.png"


@app.post("/api/reviewed-masks")
async def save_reviewed_mask(request: Request):
    """Save a reviewed mask as a separate PNG and JSON record."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Request body must be valid JSON.",
        ) from exc

    filename = str(payload.get("filename") or "uploaded_image")
    experiment_id = payload.get("experiment_id")
    encoded_mask = payload.get("mask_png_base64")
    width = payload.get("width")
    height = payload.get("height")

    if not isinstance(encoded_mask, str) or not encoded_mask:
        raise HTTPException(
            status_code=400,
            detail="A base64-encoded PNG mask is required.",
        )

    # Validate dimensions before decoding/allocating large arrays.
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width < 1
        or height < 1
        or width > 10000
        or height > 10000
        or width * height > 40_000_000
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid mask dimensions.",
        )

    try:
        png_bytes = base64.b64decode(encoded_mask, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Mask data is not valid base64.",
        ) from exc

    if len(png_bytes) > 30 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Reviewed mask is too large.",
        )

    # Decode and verify the file rather than trusting its extension.
    mask_array = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask_array is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the reviewed mask PNG.",
        )

    actual_height, actual_width = mask_array.shape[:2]
    if actual_width != width or actual_height != height:
        raise HTTPException(
            status_code=400,
            detail="Mask dimensions do not match the submitted dimensions.",
        )

    # Persist a strict binary mask: foreground=255, background=0.
    binary_mask = np.where(mask_array > 127, 255, 0).astype(np.uint8)
    success, encoded_png = cv2.imencode(".png", binary_mask)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode the reviewed mask.",
        )

    review_id = str(uuid.uuid4())
    mask_path = _review_mask_path(review_id)
    metadata_path = _review_metadata_path(review_id)

    metadata = {
        "review_id": review_id,
        "filename": Path(filename).name,
        "experiment_id": (
            str(experiment_id) if experiment_id else None
        ),
        "width": actual_width,
        "height": actual_height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mask_file": mask_path.name,
        "status": "reviewed",
        "note": (
            "Human-reviewed annotation saved separately from "
            "the original AI prediction."
        ),
    }

    try:
        mask_path.write_bytes(encoded_png.tobytes())
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        # Avoid leaving a partial record if either write fails.
        mask_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not save reviewed annotation: {exc}",
        ) from exc

    return {
        **metadata,
        "mask_url": f"/api/reviewed-masks/{review_id}/mask",
    }


@app.get("/api/reviewed-masks")
def list_reviewed_masks():
    """List saved human-reviewed annotation records."""
    records = []

    for metadata_path in REVIEWED_MASKS_DIR.glob("*.json"):
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            review_id = metadata.get("review_id")

            if not review_id or not _review_mask_path(review_id).is_file():
                continue

            records.append({
                **metadata,
                "mask_url": f"/api/reviewed-masks/{review_id}/mask",
            })
        except (OSError, ValueError, TypeError):
            continue

    records.sort(
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )

    return {"total": len(records), "reviews": records}


@app.get("/api/reviewed-masks/{review_id}")
def get_reviewed_mask_metadata(review_id: str):
    """Retrieve metadata for one saved reviewed annotation."""
    try:
        parsed_id = str(uuid.UUID(review_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid review ID.",
        ) from exc

    metadata_path = _review_metadata_path(parsed_id)
    mask_path = _review_mask_path(parsed_id)

    if not metadata_path.is_file() or not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Reviewed annotation not found.",
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read reviewed annotation metadata.",
        ) from exc

    return {
        **metadata,
        "mask_url": f"/api/reviewed-masks/{parsed_id}/mask",
    }


@app.get("/api/reviewed-masks/{review_id}/mask")
def get_reviewed_mask_file(review_id: str):
    """Download a saved reviewed mask as a PNG."""
    try:
        parsed_id = str(uuid.UUID(review_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid review ID.",
        ) from exc

    mask_path = _review_mask_path(parsed_id)

    if not mask_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Reviewed mask not found.",
        )

    return FileResponse(
        path=str(mask_path),
        media_type="image/png",
        filename=f"reviewed-{parsed_id}.png",
    )


@app.get("/api/reviewed-masks/{review_id}/evaluate")
def evaluate_reviewed_mask(review_id: str):
    """
    Compare the original AI prediction and reviewed mask
    against the same BUSI ground-truth mask.
    """
    import json
    import uuid

    # Validate review ID.
    try:
        review_id = str(uuid.UUID(review_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid review ID.",
        ) from exc

    metadata_path = REVIEWED_MASKS_DIR / f"{review_id}.json"
    reviewed_path = REVIEWED_MASKS_DIR / f"{review_id}.png"

    if not metadata_path.is_file() or not reviewed_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Reviewed annotation not found.",
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read review metadata.",
        ) from exc

    experiment_id = metadata.get("experiment_id")

    if not experiment_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "This review is not linked to an experiment. "
                "Metrics require the original input and AI prediction."
            ),
        )

    experiment = get_experiment(str(experiment_id))

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Linked experiment was not found.",
        )

    input_path = get_artifact_path(str(experiment_id), "input")
    prediction_path = get_artifact_path(
        str(experiment_id), "prediction"
    )

    if input_path is None or prediction_path is None:
        raise HTTPException(
            status_code=404,
            detail="Original input or prediction artifact is missing.",
        )

    # Load the original input image.
    input_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)

    # Load both masks as grayscale images.
    ai_mask = cv2.imread(
        str(prediction_path), cv2.IMREAD_GRAYSCALE
    )
    reviewed_mask = cv2.imread(
        str(reviewed_path), cv2.IMREAD_GRAYSCALE
    )

    if input_bgr is None or ai_mask is None or reviewed_mask is None:
        raise HTTPException(
            status_code=500,
            detail="Could not decode one or more image artifacts.",
        )

    if (
        ai_mask.shape != reviewed_mask.shape
        or ai_mask.shape != input_bgr.shape[:2]
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "The original image, AI mask, and reviewed mask "
                "must have matching dimensions."
            ),
        )

    # Find the ground truth only if the original image matches BUSI.
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

    ground_truth, ground_truth_path = find_busi_ground_truth(
        experiment["filename"], input_rgb
    )

    if ground_truth is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No exact, unambiguous BUSI ground-truth match "
                "was found. Metrics were not calculated."
            ),
        )

    # BUSI masks are binary. Convert every mask to foreground/background.
    ai_binary = np.where(ai_mask > 127, 255, 0).astype(np.uint8)
    reviewed_binary = np.where(
        reviewed_mask > 127, 255, 0
    ).astype(np.uint8)
    gt_binary = np.where(
        ground_truth > 0, 255, 0
    ).astype(np.uint8)

    # Calculate both against the SAME ground-truth mask.
    ai_metrics = calculate_segmentation_metrics(
        ai_binary, gt_binary
    )
    reviewed_metrics = calculate_segmentation_metrics(
        reviewed_binary, gt_binary
    )

    metric_names = ["dice", "iou", "precision", "recall"]

    changes = {
        name: round(
            reviewed_metrics[name] - ai_metrics[name], 6
        )
        for name in metric_names
    }

    return {
        "review_id": review_id,
        "experiment_id": str(experiment_id),
        "filename": experiment["filename"],
        "ground_truth_file": ground_truth_path.name,
        "original_ai_metrics": ai_metrics,
        "reviewed_metrics": reviewed_metrics,
        "changes": changes,
        "interpretation": (
            "Metrics compare each mask with the same BUSI "
            "ground truth. Positive changes indicate higher "
            "scores for the reviewed mask."
        ),
    }

# ============================================================
# RESEARCH REPORT GENERATOR: PDF REPORTS AND CSV EXPORT
# These endpoints read saved artifacts; they do not rerun inference.
# Requires: pip install reportlab
# ============================================================


def _latest_review_for_experiment(experiment_id: str):
    """Return the newest saved review linked to an experiment, if any."""
    records = []
    for metadata_path in REVIEWED_MASKS_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("experiment_id") or "") != str(experiment_id):
                continue
            review_id = metadata.get("review_id")
            if not review_id:
                continue
            mask_path = _review_mask_path(str(review_id))
            if mask_path.is_file():
                metadata["mask_path"] = mask_path
                records.append(metadata)
        except (OSError, ValueError, TypeError):
            continue
    records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return records[0] if records else None


def _format_report_value(value, digits: int = 4) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def _report_image_from_array(array: np.ndarray, max_width=165 * mm, max_height=95 * mm):
    """Convert an array to a report-sized in-memory ReportLab image."""
    image = Image.fromarray(array)
    image.thumbnail((int(max_width / mm * 3.2), int(max_height / mm * 3.2)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ReportImage(buffer, width=min(max_width, image.width * 0.75), height=min(max_height, image.height * 0.75))


def _report_metrics_table(metrics: dict | None):
    metrics = metrics or {}
    rows = [["Metric", "Value"]]
    for key, label in (("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")):
        value = metrics.get(key)
        rows.append([label, "Not available" if value is None else f"{float(value):.4f}"])
    table = Table(rows, colWidths=[65 * mm, 65 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DCE3")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F9FB")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


@app.get("/api/experiments/{experiment_id}/report.pdf")
def download_experiment_report(experiment_id: str):
    """Create a PDF report from a saved experiment and its latest review."""
    experiment = get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    input_path = get_artifact_path(experiment_id, "input")
    prediction_path = get_artifact_path(experiment_id, "prediction")
    if input_path is None or prediction_path is None:
        raise HTTPException(
            status_code=409,
            detail="A PDF report requires a saved input image and prediction mask.",
        )

    input_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    ai_mask = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    if input_bgr is None or ai_mask is None:
        raise HTTPException(status_code=500, detail="Could not decode saved report artifacts.")
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    if ai_mask.shape != input_rgb.shape[:2]:
        ai_mask = cv2.resize(ai_mask, (input_rgb.shape[1], input_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    review = _latest_review_for_experiment(experiment_id)
    reviewed_mask = None
    review_evaluation = None
    if review:
        reviewed_mask = cv2.imread(str(review["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if reviewed_mask is not None and reviewed_mask.shape != ai_mask.shape:
            reviewed_mask = cv2.resize(reviewed_mask, (ai_mask.shape[1], ai_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            review_evaluation = evaluate_reviewed_mask(str(review["review_id"]))
        except Exception:
            # A review may be saved for an image without a matching BUSI ground truth.
            review_evaluation = None

    # Make an overlay that distinguishes AI and reviewed regions.
    overlay = input_rgb.copy()
    ai_foreground = ai_mask > 127
    overlay[ai_foreground] = (
        0.60 * overlay[ai_foreground] + 0.40 * np.array([255, 70, 70])
    ).astype(np.uint8)
    if reviewed_mask is not None:
        reviewed_foreground = reviewed_mask > 127
        overlay[reviewed_foreground] = (
            0.60 * overlay[reviewed_foreground] + 0.40 * np.array([55, 210, 120])
        ).astype(np.uint8)

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"MedSegment AI Report - {experiment_id}",
        author="MedSegment AI",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.HexColor("#526273"), spaceAfter=10))
    styles.add(ParagraphStyle(name="SmallReport", parent=styles["BodyText"], fontSize=8, leading=10, wordWrap="CJK"))
    story = [
        Paragraph("MedSegment AI — Segmentation Experiment Report", styles["Title"]),
        Paragraph(f"Experiment ID: {html.escape(experiment_id)}", styles["ReportSubtitle"]),
        Spacer(1, 4 * mm),
        Paragraph("1. Experiment Summary", styles["Heading2"]),
    ]
    summary_rows = [["Field", "Value"]]
    summary_fields = [
        ("Created at (UTC)", experiment.get("created_at")),
        ("Status", experiment.get("status")),
        ("Input filename", experiment.get("filename")),
        ("Model", experiment.get("model")),
        ("Device", experiment.get("device")),
        ("Inference time (seconds)", _format_report_value(experiment.get("inference_seconds"))),
        ("MedSAM score", _format_report_value(experiment.get("medsam_score"))),
        ("Predicted bounding box (x1, y1, x2, y2)", _format_report_value(experiment.get("bbox"))),
        ("Prompt checkpoint", experiment.get("prompt_checkpoint")),
        ("MedSAM checkpoint", experiment.get("medsam_checkpoint")),
    ]
    for key, value in summary_fields:
        summary_rows.append([key, Paragraph(html.escape(_format_report_value(value)), styles["SmallReport"])])
    summary_table = Table(summary_rows, colWidths=[55 * mm, 112 * mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DCE3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [summary_table, Spacer(1, 5 * mm), Paragraph("2. Model Configuration", styles["Heading2"])]
    config_text = html.escape(json.dumps(experiment.get("config") or {}, indent=2, ensure_ascii=False))
    story.append(Paragraph(config_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["SmallReport"]))
    story += [Spacer(1, 4 * mm), Paragraph("3. Saved Segmentation Artifacts", styles["Heading2"])]

    def add_artifact(title, array, cmap=None):
        story.append(Paragraph(title, styles["Heading3"]))
        if array is None:
            story.append(Paragraph("Not available.", styles["Normal"]))
            return
        story.append(_report_image_from_array(array))
        story.append(Spacer(1, 3 * mm))

    add_artifact("Original ultrasound image", input_rgb)
    add_artifact("AI-predicted binary mask", np.where(ai_mask > 127, 255, 0).astype(np.uint8))
    if reviewed_mask is not None:
        add_artifact("Human-reviewed binary mask", np.where(reviewed_mask > 127, 255, 0).astype(np.uint8))
    else:
        add_artifact("Human-reviewed binary mask", None)
    add_artifact("Overlay (AI regions in red; reviewed regions in green when present)", overlay)

    story += [PageBreak(), Paragraph("4. Segmentation Metrics", styles["Heading2"])]
    ai_metrics = {key: experiment.get(key) for key in ("dice", "iou", "precision", "recall")}
    story.append(Paragraph("Stored AI prediction metrics. These values are only populated when a valid ground-truth comparison was available.", styles["BodyText"]))
    story.append(_report_metrics_table(ai_metrics))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("5. Review Details", styles["Heading2"]))
    if review:
        review_rows = [
            ["Review ID", str(review.get("review_id", "Not available"))],
            ["Review created at (UTC)", str(review.get("created_at", "Not available"))],
            ["Review status", str(review.get("status", "reviewed"))],
            ["Review note", str(review.get("note", "Not available"))],
        ]
        review_table = Table([[html.escape(str(k)), Paragraph(html.escape(str(v)), styles["SmallReport"])] for k, v in review_rows], colWidths=[55 * mm, 112 * mm])
        review_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DCE3")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 5)]))
        story.append(review_table)
        if review_evaluation:
            story += [Spacer(1, 4 * mm), Paragraph("AI versus reviewed mask (each compared with the same BUSI ground truth)", styles["Heading3"])]
            compare_rows = [["Metric", "AI prediction", "Reviewed mask", "Change"]]
            for key in ("dice", "iou", "precision", "recall"):
                compare_rows.append([
                    key.capitalize(),
                    f"{review_evaluation['original_ai_metrics'][key]:.4f}",
                    f"{review_evaluation['reviewed_metrics'][key]:.4f}",
                    f"{review_evaluation['changes'][key]:+.4f}",
                ])
            compare_table = Table(compare_rows, colWidths=[40 * mm, 42 * mm, 42 * mm, 42 * mm], repeatRows=1)
            compare_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243447")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DCE3")), ("PADDING", (0, 0), (-1, -1), 5)]))
            story.append(compare_table)
            story.append(Paragraph("Ground-truth file: " + html.escape(str(review_evaluation.get("ground_truth_file", "Not available"))), styles["SmallReport"]))
        else:
            story.append(Paragraph("Reviewed-mask metrics are unavailable because a valid exact BUSI ground-truth match could not be established.", styles["BodyText"]))
    else:
        story.append(Paragraph("No reviewed mask is linked to this experiment.", styles["BodyText"]))

    story += [Spacer(1, 6 * mm), Paragraph("6. Interpretation and Limitations", styles["Heading2"])]
    story.append(Paragraph(
        "This report documents an experimental segmentation run and saved artifacts. Metrics are meaningful only when computed against an appropriate ground-truth mask. A human-reviewed mask is an annotation artifact and is not itself proof of clinical correctness. MedSegment AI is a research prototype and must not be used as a medical diagnosis.",
        styles["BodyText"],
    ))
    if experiment.get("error"):
        story += [Spacer(1, 3 * mm), Paragraph("Recorded error: " + html.escape(str(experiment["error"])), styles["BodyText"])]

    try:
        doc.build(story)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not generate PDF report: {exc}") from exc
    pdf_buffer.seek(0)
    safe_id = "".join(ch for ch in experiment_id if ch.isalnum() or ch in "-_" )
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="medsegment-report-{safe_id}.pdf"'},
    )


@app.get("/api/experiments/export/history.csv")
def export_experiment_history_csv():
    """Export saved experiment history and latest linked review details as CSV."""
    experiments = list_experiments(500)
    reviews_by_experiment = {}
    for metadata_path in REVIEWED_MASKS_DIR.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            experiment_id = metadata.get("experiment_id")
            review_id = metadata.get("review_id")
            if not experiment_id or not review_id or not _review_mask_path(str(review_id)).is_file():
                continue
            current = reviews_by_experiment.get(str(experiment_id))
            if current is None or metadata.get("created_at", "") > current.get("created_at", ""):
                reviews_by_experiment[str(experiment_id)] = metadata
        except (OSError, ValueError, TypeError):
            continue

    output = io.StringIO(newline="")
    columns = [
        "id", "created_at", "status", "filename", "model", "device",
        "prompt_checkpoint", "medsam_checkpoint", "inference_seconds", "medsam_score",
        "dice", "iou", "precision", "recall", "bbox", "config", "error",
        "review_id", "review_created_at", "review_status", "review_note",
    ]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for experiment in experiments:
        review = reviews_by_experiment.get(str(experiment.get("id")), {})
        row = dict(experiment)
        row["bbox"] = json.dumps(experiment.get("bbox") or [], ensure_ascii=False)
        row["config"] = json.dumps(experiment.get("config") or {}, ensure_ascii=False)
        row["review_id"] = review.get("review_id", "")
        row["review_created_at"] = review.get("created_at", "")
        row["review_status"] = review.get("status", "")
        row["review_note"] = review.get("note", "")
        writer.writerow(row)
    csv_bytes = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    return StreamingResponse(
        csv_bytes,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="medsegment-experiment-history.csv"'},
    )

