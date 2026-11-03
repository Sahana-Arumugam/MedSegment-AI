from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
METRICS = ROOT / "backend/box_coordinate_diagnostics/box_coordinate_metrics.csv"
MANIFEST = ROOT / "backend/prepared_dataset/busi_manifest.csv"
OUT = ROOT / "backend/box_coordinate_diagnostics"

metrics = pd.read_csv(METRICS)
manifest = pd.read_csv(MANIFEST)

# Match images by filename.
manifest["filename"] = manifest["image_path"].apply(
    lambda p: Path(str(p)).name
)

gt_cols = [
    "filename", "width", "height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
]
gt = manifest[gt_cols].drop_duplicates("filename")

df = metrics.merge(gt, on="filename", how="left", validate="many_to_one")

if df[["width", "height", "bbox_x1", "bbox_y1",
       "bbox_x2", "bbox_y2"]].isna().any().any():
    raise ValueError("Some evaluated images could not be matched to GT boxes.")

# Predicted and ground-truth box geometry.
df["pred_cx"] = (df["pred_x1"] + df["pred_x2"]) / 2
df["pred_cy"] = (df["pred_y1"] + df["pred_y2"]) / 2
df["pred_w"] = df["pred_x2"] - df["pred_x1"]
df["pred_h"] = df["pred_y2"] - df["pred_y1"]

df["gt_cx"] = (df["bbox_x1"] + df["bbox_x2"]) / 2
df["gt_cy"] = (df["bbox_y1"] + df["bbox_y2"]) / 2
df["gt_w"] = df["bbox_x2"] - df["bbox_x1"]
df["gt_h"] = df["bbox_y2"] - df["bbox_y1"]

# Normalize center errors by image dimensions.
df["center_error_x_norm"] = (
    (df["pred_cx"] - df["gt_cx"]).abs() / df["width"]
)
df["center_error_y_norm"] = (
    (df["pred_cy"] - df["gt_cy"]).abs() / df["height"]
)
df["center_error_norm"] = np.sqrt(
    df["center_error_x_norm"]**2 + df["center_error_y_norm"]**2
)

df["width_ratio"] = df["pred_w"] / df["gt_w"].clip(lower=1)
df["height_ratio"] = df["pred_h"] / df["gt_h"].clip(lower=1)

# Save per-image diagnostic measurements separately.
detail_path = OUT / "box_error_geometry_analysis.csv"
df.to_csv(detail_path, index=False)

print("\n===== BOX ERROR SUMMARY BY MODEL =====")
summary = df.groupby("model").agg(
    images=("filename", "nunique"),
    mean_box_iou=("bbox_iou_vs_mask_bbox", "mean"),
    median_center_error=("center_error_norm", "median"),
    mean_abs_x_error=("center_error_x_norm", "mean"),
    mean_abs_y_error=("center_error_y_norm", "mean"),
    median_width_ratio=("width_ratio", "median"),
    median_height_ratio=("height_ratio", "median"),
)
print(summary.round(4).to_string())

print("\n===== CENTER ERROR BY LESION SIZE =====")
by_size = df.groupby(["model", "lesion_size"]).agg(
    images=("filename", "nunique"),
    mean_box_iou=("bbox_iou_vs_mask_bbox", "mean"),
    median_center_error=("center_error_norm", "median"),
    median_width_ratio=("width_ratio", "median"),
    median_height_ratio=("height_ratio", "median"),
)
print(by_size.round(4).to_string())

print("\n===== WORST V3 CENTER-LOCALIZATION ERRORS =====")
v3 = df[df["model"] == "V3"].sort_values(
    "center_error_norm", ascending=False
)
print(v3[[
    "filename", "lesion_size", "center_error_x_norm",
    "center_error_y_norm", "center_error_norm",
    "width_ratio", "height_ratio",
    "bbox_iou_vs_mask_bbox", "dice"
]].head(10).round(4).to_string(index=False))

print(f"\nDetailed results saved to: {detail_path}")
print("Read-only diagnostic: no model weights changed.")
