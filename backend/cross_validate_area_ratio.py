from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path(
    "backend/prompt_stability_validation/geometry_analysis.csv"
)
OUTPUT = Path(
    "backend/prompt_stability_validation/area_ratio_cv_results.csv"
)

df = pd.read_csv(SOURCE).reset_index(drop=True)

required = [
    "base_mask_area_box_fraction",
    "base_dice",
    "expanded_dice",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).reset_index(drop=True)
if len(df) < 20:
    raise ValueError("Too few complete records for 5-fold evaluation.")

# The feature is measured from the original predicted mask and box.
x = df["base_mask_area_box_fraction"].to_numpy(dtype=float)

# Ground-truth Dice is used only to train/evaluate the offline rule.
base = df["base_dice"].to_numpy(dtype=float)
expanded = df["expanded_dice"].to_numpy(dtype=float)
n = len(df)

# Deterministic shuffled 5-fold split.
rng = np.random.default_rng(42)
indices = rng.permutation(n)
fold_ids = np.empty(n, dtype=int)
for j, chunk in enumerate(np.array_split(indices, 5)):
    fold_ids[chunk] = j

oof = np.full(n, np.nan)
chosen_threshold = np.full(n, np.nan)
chosen_direction = np.full(n, "", dtype=object)
fold_summary = []

def choose_expansion(feature, threshold, direction):
    if direction == "low":
        return feature <= threshold
    return feature >= threshold

for fold in range(5):
    test_idx = np.where(fold_ids == fold)[0]
    train_idx = np.where(fold_ids != fold)[0]

    # Search thresholds using training folds only.
    unique_values = np.unique(x[train_idx])
    if len(unique_values) == 1:
        thresholds = unique_values
    else:
        thresholds = (
            unique_values[:-1] + unique_values[1:]
        ) / 2

    best_score = -np.inf
    best_threshold = None
    best_direction = None

    for threshold in thresholds:
        for direction in ("low", "high"):
            expand_train = choose_expansion(
                x[train_idx], threshold, direction
            )
            predicted_train = np.where(
                expand_train, expanded[train_idx], base[train_idx]
            )
            score = predicted_train.mean()

            # Deterministic tie handling: keep first best rule.
            if score > best_score + 1e-12:
                best_score = score
                best_threshold = float(threshold)
                best_direction = direction

    expand_test = choose_expansion(
        x[test_idx], best_threshold, best_direction
    )
    oof[test_idx] = np.where(
        expand_test, expanded[test_idx], base[test_idx]
    )
    chosen_threshold[test_idx] = best_threshold
    chosen_direction[test_idx] = best_direction

    fold_summary.append({
        "fold": fold + 1,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "threshold": best_threshold,
        "direction": best_direction,
        "train_rule_dice": best_score,
        "test_selector_dice": oof[test_idx].mean(),
        "test_base_dice": base[test_idx].mean(),
        "test_expand_dice": expanded[test_idx].mean(),
    })

# Save per-image out-of-fold predictions in a new file.
out = df[["filename"]].copy()
out["fold"] = fold_ids + 1
out["area_ratio"] = x
out["base_dice"] = base
out["expanded_dice"] = expanded
out["selected_dice_oof"] = oof
out["threshold_learned_without_this_fold"] = chosen_threshold
out["rule_direction"] = chosen_direction
out["selector_gain_vs_base"] = oof - base
out["selector_gain_vs_expansion"] = oof - expanded
out.to_csv(OUTPUT, index=False)

print("\n===== 5-FOLD OUT-OF-FOLD RESULTS =====")
print("Images:", n)
print("Folds: 5")
print("Original mean Dice:", round(base.mean(), 4))
print("Always expand mean Dice:", round(expanded.mean(), 4))
print("Area-ratio selector OOF mean Dice:", round(oof.mean(), 4))
print("Selector gain vs original:", round((oof - base).mean(), 4))
print("Selector gain vs always expand:", round((oof - expanded).mean(), 4))
print("Images selector beats original:", int((oof > base + 1e-9).sum()))
print("Images selector loses to original:", int((oof < base - 1e-9).sum()))
print("Images selector beats expansion:", int((oof > expanded + 1e-9).sum()))
print("Images selector loses to expansion:", int((oof < expanded - 1e-9).sum()))

print("\n===== FOLD DETAILS =====")
print(pd.DataFrame(fold_summary).round(4).to_string(index=False))

print("\n===== RULE CONSISTENCY =====")
print("Chosen directions:")
print(pd.Series(chosen_direction).value_counts().to_string())
print("Learned thresholds by fold:",
      [round(s["threshold"], 4) for s in fold_summary])

print("\nSaved per-image results:", OUTPUT)
print("Original geometry CSV was not modified.")
