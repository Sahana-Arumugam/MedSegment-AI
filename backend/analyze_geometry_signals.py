from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

path = Path("backend/prompt_stability_validation/geometry_analysis.csv")
df = pd.read_csv(path)

candidates = ["base", "expanded", "contracted"]
dice_cols = [f"{c}_dice" for c in candidates]
dice = df[dice_cols].to_numpy()

# Label the prompt that achieved the highest Dice.
winner_idx = np.argmax(dice, axis=1)
df["best_prompt"] = [candidates[i] for i in winner_idx]

print("\n===== 1. BEST PROMPT COUNTS =====")
print(df["best_prompt"].value_counts().reindex(candidates, fill_value=0).to_string())
print("\nMean Dice by fixed strategy:")
for c in candidates:
    print(f"{c}: {df[f'{c}_dice'].mean():.4f}")

print("\n===== 2. GEOMETRY SIGNAL VS DICE =====")
features = [
    "mask_area_image_fraction",
    "mask_area_box_fraction",
    "connected_components",
    "largest_component_fraction",
    "mask_fraction_outside_box",
    "mask_touches_box_boundary",
]

results = []
for c in candidates:
    for feature in features:
        col = f"{c}_{feature}"
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        y = df[f"{c}_dice"]
        valid = x.notna() & y.notna()
        if valid.sum() < 5 or x[valid].nunique() < 2:
            continue
        rho = spearmanr(x[valid], y[valid]).statistic
        results.append({
            "prompt": c,
            "feature": feature,
            "spearman_vs_dice": rho,
        })

result_df = pd.DataFrame(results)
if not result_df.empty:
    result_df["abs_rho"] = result_df["spearman_vs_dice"].abs()
    print(result_df.sort_values("abs_rho", ascending=False)
          .drop(columns="abs_rho").round(3).to_string(index=False))

print("\n===== 3. GEOMETRY BY ORACLE WINNER =====")
group_features = [
    f"{c}_{feature}"
    for c in candidates
    for feature in features
    if f"{c}_{feature}" in df.columns
]
summary = df.groupby("best_prompt")[group_features].mean(numeric_only=True)
print(summary.round(4).to_string())

print("\n===== 4. ORACLE UPPER BOUND =====")
print("Oracle mean Dice:", round(dice.max(axis=1).mean(), 4))
print("Original mean Dice:", round(df["base_dice"].mean(), 4))
print("Expanded mean Dice:", round(df["expanded_dice"].mean(), 4))
print("Contracted mean Dice:", round(df["contracted_dice"].mean(), 4))

print("\nAnalysis complete. Source CSV was not modified.")
