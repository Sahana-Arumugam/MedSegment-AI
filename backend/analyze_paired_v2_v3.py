from pathlib import Path
import pandas as pd
import numpy as np

source = Path("backend/box_coordinate_diagnostics/box_coordinate_metrics.csv")
output = Path("backend/box_coordinate_diagnostics/paired_v2_v3_analysis.csv")

df = pd.read_csv(source)

required = {"model", "filename", "lesion_size", "dice",
            "bbox_iou_vs_mask_bbox", "segmentation_iou"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.duplicated(["model", "filename"]).any():
    raise ValueError("Duplicate model/image rows found.")

wide = df.pivot(
    index=["filename", "lesion_size"],
    columns="model",
    values=["dice", "bbox_iou_vs_mask_bbox", "segmentation_iou"]
)

if not {"V2", "V3"}.issubset(set(df["model"].unique())):
    raise ValueError("Both V2 and V3 results are required.")

wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
wide = wide.reset_index()

for metric in ["dice", "bbox_iou_vs_mask_bbox", "segmentation_iou"]:
    wide[f"{metric}_delta"] = wide[f"{metric}_V3"] - wide[f"{metric}_V2"]

if wide[["dice_V2", "dice_V3"]].isna().any().any():
    raise ValueError("Some images do not have results from both models.")

def summarize(group):
    result = {"images": len(group)}
    for metric in ["dice", "bbox_iou_vs_mask_bbox", "segmentation_iou"]:
        delta = group[f"{metric}_delta"]
        result[f"V2_{metric}_mean"] = group[f"{metric}_V2"].mean()
        result[f"V3_{metric}_mean"] = group[f"{metric}_V3"].mean()
        result[f"{metric}_mean_delta"] = delta.mean()
        result[f"{metric}_median_delta"] = delta.median()
        result[f"{metric}_improved"] = int((delta > 1e-9).sum())
        result[f"{metric}_worsened"] = int((delta < -1e-9).sum())
        result[f"{metric}_unchanged"] = int((delta.abs() <= 1e-9).sum())
    return pd.Series(result)

print("\n===== PAIRED V2 VS V3: ALL IMAGES =====")
print(summarize(wide).round(4).to_string())

print("\n===== PAIRED RESULTS BY LESION SIZE =====")
print(
    wide.groupby("lesion_size", sort=False)
        .apply(summarize, include_groups=False)
        .round(4).to_string()
)

print("\n===== LARGEST DICE IMPROVEMENTS FROM V3 =====")
print(
    wide.sort_values("dice_delta", ascending=False)
        [["filename", "lesion_size", "dice_V2", "dice_V3",
          "dice_delta", "bbox_iou_vs_mask_bbox_delta"]]
        .head(10).round(4).to_string(index=False)
)

print("\n===== LARGEST DICE REGRESSIONS FROM V3 =====")
print(
    wide.sort_values("dice_delta")
        [["filename", "lesion_size", "dice_V2", "dice_V3",
          "dice_delta", "bbox_iou_vs_mask_bbox_delta"]]
        .head(10).round(4).to_string(index=False)
)

wide.to_csv(output, index=False)
print(f"\nPaired per-image results saved to: {output}")
