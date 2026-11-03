import pandas as pd
import numpy as np

path = "backend/prompt_stability_validation/validation_prompt_stability.csv"
df = pd.read_csv(path)

base = df["base_dice"]
expanded = df["expanded_dice"]
contracted = df["contracted_dice"]

delta = expanded - base

print("\n===== BOX EXPANSION ANALYSIS =====")
print("Mean Dice change:", round(delta.mean(), 4))
print("Median Dice change:", round(delta.median(), 4))
print("Improved:", int((delta > 0).sum()))
print("Unchanged:", int((delta == 0).sum()))
print("Worsened:", int((delta < 0).sum()))
print("Improvement >= 0.05:", int((delta >= 0.05).sum()))
print("Worsening <= -0.05:", int((delta <= -0.05).sum()))

print("\n===== BEST FIXED STRATEGY =====")
results = {
    "Base": base.mean(),
    "Expand 10%": expanded.mean(),
    "Contract 10%": contracted.mean(),
}
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")

print("\n===== LARGEST EXPANSION IMPROVEMENTS =====")
cols = ["filename", "base_dice", "expanded_dice", "contracted_dice"]
print(
    df.assign(delta=delta)
      .sort_values("delta", ascending=False)
      [["filename", "base_dice", "expanded_dice", "contracted_dice", "delta"]]
      .head(10).round(4).to_string(index=False)
)

print("\n===== LARGEST EXPANSION REGRESSIONS =====")
print(
    df.assign(delta=delta)
      .sort_values("delta")
      [["filename", "base_dice", "expanded_dice", "contracted_dice", "delta"]]
      .head(10).round(4).to_string(index=False)
)
