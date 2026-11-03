from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GEOMETRY = ROOT / "prompt_stability_validation" / "geometry_analysis.csv"
PREDICTIONS = ROOT / "failure_detector_cv" / "oof_failure_predictions.csv"
OUT_DIR = ROOT / "failure_detector_cv"
OUTPUT = OUT_DIR / "selective_refinement_results.csv"

geo = pd.read_csv(GEOMETRY)
pred = pd.read_csv(PREDICTIONS)

df = geo.merge(
    pred[["filename", "failure_probability_oof"]],
    on="filename",
    how="inner",
    validate="one_to_one",
).dropna(subset=[
    "base_dice", "expanded_dice", "failure_probability_oof"
]).reset_index(drop=True)

if len(df) != len(pred):
    raise ValueError("Unexpected unmatched rows; inspect filenames before proceeding.")

base = df["base_dice"].to_numpy(float)
expanded = df["expanded_dice"].to_numpy(float)
risk = df["failure_probability_oof"].to_numpy(float)

# Thresholds are fixed in advance; no Dice-based threshold tuning.
policies = {
    "original_v2": np.zeros(len(df), dtype=bool),
    "always_expand_10pct": np.ones(len(df), dtype=bool),
    "detector_threshold_0.50": risk >= 0.50,
    "detector_top_25pct_risk": risk >= np.quantile(risk, 0.75),
    "detector_top_50pct_risk": risk >= np.quantile(risk, 0.50),
}

rows = []
for name, expand in policies.items():
    selected_dice = np.where(expand, expanded, base)
    rows.append({
        "policy": name,
        "images": len(df),
        "expanded_images": int(expand.sum()),
        "expansion_coverage": float(expand.mean()),
        "mean_dice": float(selected_dice.mean()),
        "median_dice": float(np.median(selected_dice)),
        "gain_vs_original": float((selected_dice - base).mean()),
        "gain_vs_always_expand": float((selected_dice - expanded).mean()),
        "poor_cases_dice_below_0_30": int((selected_dice < 0.30).sum()),
    })

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print("\n===== SELECTIVE REFINEMENT COMPARISON =====")
print(result.round(4).to_string(index=False))
print("\nSaved:", OUTPUT)
print("No checkpoints or existing result files were modified.")
