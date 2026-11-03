from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "prompt_stability_validation" / "geometry_analysis.csv"
OUT_DIR = ROOT / "failure_detector_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# Ground-truth Dice is used ONLY to create labels and evaluate predictions.
y = (df["base_dice"].to_numpy() < 0.30).astype(int)

# Every feature below is derived from model predictions, not ground truth.
feature_cols = [
    f"{prompt}_{feature}"
    for prompt in ["base", "expanded", "contracted"]
    for feature in [
        "medsam_score",
        "mask_area_image_fraction",
        "mask_area_box_fraction",
        "connected_components",
        "largest_component_fraction",
        "mask_fraction_outside_box",
        "mask_touches_box_boundary",
    ]
]

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing inference-time features: {missing}")

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

# Add differences between expanded/contracted and base predictions.
extra = []
for feature in [
    "medsam_score",
    "mask_area_image_fraction",
    "mask_area_box_fraction",
    "largest_component_fraction",
    "mask_fraction_outside_box",
]:
    base = df[f"base_{feature}"].to_numpy(dtype=float)
    expanded = df[f"expanded_{feature}"].to_numpy(dtype=float)
    contracted = df[f"contracted_{feature}"].to_numpy(dtype=float)
    extra.extend([expanded - base, contracted - base])

X = np.column_stack([X, *extra])

if len(df) < 30 or len(np.unique(y)) < 2:
    raise ValueError("Insufficient samples or only one failure class.")

# Stratified out-of-fold predictions. Preprocessing is fitted within each fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = np.zeros(len(df), dtype=float)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=3000,
            random_state=42,
        ),
    )
    model.fit(X[train_idx], y[train_idx])
    oof_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    print(f"Completed fold {fold}/5")

pred = (oof_prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

results = df[["filename", "base_dice"]].copy()
results["failure_label_dice_below_0_30"] = y
results["failure_probability_oof"] = oof_prob
results["failure_prediction_oof"] = pred

results_path = OUT_DIR / "oof_failure_predictions.csv"
results.to_csv(results_path, index=False)

print("\n===== OUT-OF-FOLD FAILURE DETECTION =====")
print("Images:", len(df))
print("Poor baseline predictions (Dice < 0.30):", int(y.sum()))
print("Failure prevalence:", round(float(y.mean()), 4))
print("ROC AUC:", round(roc_auc_score(y, oof_prob), 4))
print("Average precision:", round(average_precision_score(y, oof_prob), 4))
print("Balanced accuracy at threshold 0.5:",
      round(balanced_accuracy_score(y, pred), 4))
print("Failure recall:", round(tp / (tp + fn), 4) if tp + fn else 0)
print("Failure precision:", round(tp / (tp + fp), 4) if tp + fp else 0)
print("Specificity:", round(tn / (tn + fp), 4) if tn + fp else 0)
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print([[int(tn), int(fp)], [int(fn), int(tp)]])
print("\nSaved:", results_path)
print("Existing results and checkpoints were not modified.")
