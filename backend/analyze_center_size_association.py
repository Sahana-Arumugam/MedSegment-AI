from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("backend/box_coordinate_diagnostics")
SOURCE = ROOT / "box_error_geometry_analysis.csv"
OUT = ROOT / "center_size_error_association.csv"

df = pd.read_csv(SOURCE)

required = {
    "model", "filename", "lesion_size", "dice",
    "bbox_iou_vs_mask_bbox", "center_error_x_norm",
    "center_error_y_norm", "center_error_norm",
    "width_ratio", "height_ratio"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.duplicated(["model", "filename"]).any():
    raise ValueError("Duplicate model/image rows found.")

# Compare the same image under both models.
metrics = [
    "dice", "bbox_iou_vs_mask_bbox",
    "center_error_x_norm", "center_error_y_norm",
    "center_error_norm", "width_ratio", "height_ratio"
]

wide = df.pivot(
    index=["filename", "lesion_size"],
    columns="model",
    values=metrics
)

if not {"V2", "V3"}.issubset(set(df["model"].unique())):
    raise ValueError("Both V2 and V3 are required.")

wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
wide = wide.reset_index()

if wide.isna().any().any():
    raise ValueError("Incomplete paired rows detected.")

# Size error: distance from ideal ratio of 1.
# Log ratios treat a 2x box and a 0.5x box as equally far from ideal.
for model in ["V2", "V3"]:
    wide[f"width_size_error_{model}"] = np.abs(
        np.log(wide[f"width_ratio_{model}"].clip(lower=1e-6))
    )
    wide[f"height_size_error_{model}"] = np.abs(
        np.log(wide[f"height_ratio_{model}"].clip(lower=1e-6))
    )
    wide[f"combined_size_error_{model}"] = (
        wide[f"width_size_error_{model}"]
        + wide[f"height_size_error_{model}"]
    ) / 2

# Positive error deltas mean V3 has more error than V2.
wide["dice_delta_v3_minus_v2"] = wide["dice_V3"] - wide["dice_V2"]

for name in [
    "center_error_x_norm", "center_error_y_norm", "center_error_norm",
    "combined_size_error"
]:
    wide[f"{name}_delta_v3_minus_v2"] = (
        wide[f"{name}_V3"] - wide[f"{name}_V2"]
    )

wide["box_iou_delta_v3_minus_v2"] = (
    wide["bbox_iou_vs_mask_bbox_V3"]
    - wide["bbox_iou_vs_mask_bbox_V2"]
)

def correlations(frame, error_columns, target):
    rows = []
    for col in error_columns:
        valid = frame[[col, target]].dropna()
        rows.append({
            "error_measure": col,
            "images": len(valid),
            "pearson_r": valid[col].corr(valid[target], method="pearson"),
            "spearman_r": valid[col].corr(valid[target], method="spearman"),
        })
    return pd.DataFrame(rows)

print("\n===== V3: DICE VS BOX ERROR =====")
v3_error_cols = [
    "center_error_x_norm_V3",
    "center_error_y_norm_V3",
    "center_error_norm_V3",
    "combined_size_error_V3",
]
print(
    correlations(wide, v3_error_cols, "dice_V3")
    .round(4).to_string(index=False)
)

print("\n===== PAIRED CHANGES: V3 ERROR CHANGE VS DICE CHANGE =====")
delta_error_cols = [
    "center_error_x_norm_delta_v3_minus_v2",
    "center_error_y_norm_delta_v3_minus_v2",
    "center_error_norm_delta_v3_minus_v2",
    "combined_size_error_delta_v3_minus_v2",
    "box_iou_delta_v3_minus_v2",
]
print(
    correlations(wide, delta_error_cols, "dice_delta_v3_minus_v2")
    .round(4).to_string(index=False)
)

print("\n===== PAIRED CHANGE SUMMARY BY LESION SIZE =====")
summary = wide.groupby("lesion_size").agg(
    images=("filename", "count"),
    mean_dice_delta=("dice_delta_v3_minus_v2", "mean"),
    mean_center_error_delta=(
        "center_error_norm_delta_v3_minus_v2", "mean"
    ),
    mean_size_error_delta=(
        "combined_size_error_delta_v3_minus_v2", "mean"
    ),
    mean_box_iou_delta=("box_iou_delta_v3_minus_v2", "mean"),
)
print(summary.round(4).to_string())

wide.to_csv(OUT, index=False)
print(f"\nPer-image paired analysis saved to: {OUT}")
print("Diagnostic only; no model weights changed.")
