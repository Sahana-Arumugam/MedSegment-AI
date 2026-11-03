from pathlib import Path
import pandas as pd
import numpy as np

p = Path("backend/prompt_stability_validation/geometry_analysis.csv")
df = pd.read_csv(p)

strategies = ["base", "expanded", "contracted"]
dice = df[[f"{s}_dice" for s in strategies]].copy()
dice.columns = strategies

print("Images:", len(df))
print("\n=== STRATEGY SUMMARY ===")

for s in strategies:
    print(
        f"{s:10s} | mean={dice[s].mean():.4f} "
        f"| median={dice[s].median():.4f} "
        f"| wins={(dice[s] == dice.max(axis=1)).sum()}"
    )

print("\n=== PAIRED CHANGE FROM BASE ===")
for s in ["expanded", "contracted"]:
    delta = dice[s] - dice["base"]
    print(
        f"{s:10s} | mean change={delta.mean():+.4f} "
        f"| improved={(delta > 1e-8).sum()} "
        f"| worsened={(delta < -1e-8).sum()} "
        f"| unchanged={(np.abs(delta) <= 1e-8).sum()}"
    )

oracle = dice.max(axis=1)
print("\n=== ORACLE (DIAGNOSTIC ONLY) ===")
print(f"Mean Dice: {oracle.mean():.4f}")
print(f"Mean gap above base: {(oracle - dice['base']).mean():.4f}")
