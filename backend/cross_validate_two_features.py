from pathlib import Path
import numpy as np
import pandas as pd

GEOMETRY = Path("backend/prompt_stability_validation/geometry_analysis.csv")
STABILITY = Path("backend/prompt_stability_validation/validation_prompt_stability.csv")
OUTPUT = Path("backend/prompt_stability_validation/two_feature_cv_results.csv")

geo = pd.read_csv(GEOMETRY)
stab = pd.read_csv(STABILITY)

needed = [
    "filename",
    "base_mask_area_box_fraction",
    "base_dice",
    "expanded_dice",
]
missing = [c for c in needed if c not in geo.columns]
if missing:
    raise ValueError(f"Missing geometry columns: {missing}")

if "mean_mask_stability" not in stab.columns:
    raise ValueError("mean_mask_stability missing from stability CSV")

df = geo.merge(
    stab[["filename", "mean_mask_stability"]],
    on="filename",
    how="inner",
    validate="one_to_one",
).dropna(subset=[
    "base_mask_area_box_fraction",
    "mean_mask_stability",
    "base_dice",
    "expanded_dice",
]).reset_index(drop=True)

X = df[
    ["base_mask_area_box_fraction", "mean_mask_stability"]
].to_numpy(dtype=float)

base = df["base_dice"].to_numpy(dtype=float)
expanded = df["expanded_dice"].to_numpy(dtype=float)
y = (expanded > base + 1e-9).astype(int)

if len(df) < 30 or len(np.unique(y)) < 2:
    raise ValueError("Insufficient records or only one target class.")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def fit_logistic(X_train, y_train, steps=3000, lr=0.05, reg=0.01):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-12] = 1.0
    Z = (X_train - mean) / std
    w = np.zeros(Z.shape[1])
    b = 0.0

    for _ in range(steps):
        p = sigmoid(Z @ w + b)
        error = p - y_train
        w -= lr * ((Z.T @ error) / len(y_train) + reg * w)
        b -= lr * error.mean()

    return mean, std, w, b

def predict(X_test, fitted):
    mean, std, w, b = fitted
    return sigmoid(((X_test - mean) / std) @ w + b)

# Deterministic stratified 5-fold assignment.
rng = np.random.default_rng(42)
fold_ids = np.empty(len(df), dtype=int)
for label in (0, 1):
    ids = np.where(y == label)[0]
    rng.shuffle(ids)
    for j, idx in enumerate(ids):
        fold_ids[idx] = j % 5

oof = np.zeros(len(df))
chosen = np.zeros(len(df), dtype=bool)
probabilities = np.zeros(len(df))
fold_rows = []

for fold in range(5):
    test_idx = np.where(fold_ids == fold)[0]
    train_idx = np.where(fold_ids != fold)[0]

    model = fit_logistic(X[train_idx], y[train_idx])
    prob = predict(X[test_idx], model)
    expand = prob >= 0.5

    oof[test_idx] = np.where(
        expand, expanded[test_idx], base[test_idx]
    )
    chosen[test_idx] = expand
    probabilities[test_idx] = prob

    fold_rows.append({
        "fold": fold + 1,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "selector_dice": oof[test_idx].mean(),
        "base_dice": base[test_idx].mean(),
        "expanded_dice": expanded[test_idx].mean(),
        "selection_accuracy": np.mean(expand == y[test_idx]),
    })

out = df[["filename"]].copy()
out["area_ratio"] = X[:, 0]
out["mean_mask_stability"] = X[:, 1]
out["base_dice"] = base
out["expanded_dice"] = expanded
out["expansion_helped_actual"] = y
out["predicted_probability_expansion_helps_oof"] = probabilities
out["selector_chose_expansion_oof"] = chosen
out["selected_dice_oof"] = oof
out.to_csv(OUTPUT, index=False)

print("\n===== TWO-FEATURE SELECTOR: 5-FOLD OOF =====")
print("Images:", len(df))
print("Baseline mean Dice:", round(base.mean(), 4))
print("Always-expand mean Dice:", round(expanded.mean(), 4))
print("Selector OOF mean Dice:", round(oof.mean(), 4))
print("Gain vs baseline:", round((oof - base).mean(), 4))
print("Gain vs always-expand:", round((oof - expanded).mean(), 4))
print("Selection accuracy:", round(np.mean(chosen == y), 4))
print("\n===== FOLD DETAILS =====")
print(pd.DataFrame(fold_rows).round(4).to_string(index=False))
print("\nSaved:", OUTPUT)
