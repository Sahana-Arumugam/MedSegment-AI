from pathlib import Path
import numpy as np
import pandas as pd

path = Path("backend/prompt_stability_validation/validation_prompt_stability.csv")
df = pd.read_csv(path)

print("\n===== VALIDATION SUMMARY =====")
print("Images:", len(df))
print("Mean baseline Dice:", round(df["base_dice"].mean(), 4))
print("Mean expanded Dice:", round(df["expanded_dice"].mean(), 4))
print("Mean contracted Dice:", round(df["contracted_dice"].mean(), 4))
print("Mean mask stability:", round(df["mean_mask_stability"].mean(), 4))

print("\n===== STABILITY VS BASELINE DICE =====")
corr = df["mean_mask_stability"].corr(df["base_dice"])
print("Pearson correlation:", round(corr, 4))

print("\n===== DICE BY MASK-STABILITY QUARTILE =====")
df["stability_quartile"] = pd.qcut(
    df["mean_mask_stability"], q=4, duplicates="drop"
)
print(
    df.groupby("stability_quartile", observed=True)["base_dice"]
      .agg(["count", "mean", "median"])
      .round(4)
      .to_string()
)

print("\n===== LOWEST-STABILITY CASES =====")
print(
    df.sort_values("mean_mask_stability")
      [["filename", "base_dice", "mean_mask_stability",
        "expanded_mask_iou_vs_base", "contracted_mask_iou_vs_base"]]
      .head(10).round(4).to_string(index=False)
)

print("\n===== LOWEST-DICE CASES =====")
print(
    df.sort_values("base_dice")
      [["filename", "base_dice", "mean_mask_stability",
        "expanded_mask_iou_vs_base", "contracted_mask_iou_vs_base"]]
      .head(10).round(4).to_string(index=False)
)
