import pandas as pd
from scipy.stats import spearmanr, pearsonr

path = "backend/prompt_stability_validation/validation_prompt_stability.csv"
df = pd.read_csv(path)

print("\n===== 1. MEDSAM SCORE VS ACTUAL DICE =====")

pairs = [
    ("Original", "base_medsam_score", "base_dice"),
    ("Expanded", "expanded_medsam_score", "expanded_dice"),
    ("Contracted", "contracted_medsam_score", "contracted_dice"),
]

for name, score_col, dice_col in pairs:
    pearson = pearsonr(df[score_col], df[dice_col]).statistic
    spearman = spearmanr(df[score_col], df[dice_col]).statistic
    print(
        f"{name}: Pearson={pearson:.4f}, "
        f"Spearman={spearman:.4f}"
    )

print("\n===== 2. DOES THE HIGHEST MEDSAM SCORE PICK THE BEST BOX? =====")

score_cols = [
    "base_medsam_score",
    "expanded_medsam_score",
    "contracted_medsam_score",
]
dice_cols = ["base_dice", "expanded_dice", "contracted_dice"]

scores = df[score_cols].to_numpy()
dice = df[dice_cols].to_numpy()

selected = scores.argmax(axis=1)
rows = range(len(df))
selected_dice = dice[rows, selected]
oracle_dice = dice.max(axis=1)
baseline_dice = df["base_dice"].to_numpy()

print("Baseline mean Dice:", round(baseline_dice.mean(), 4))
print("Highest-score selector mean Dice:", round(selected_dice.mean(), 4))
print("Oracle mean Dice (ground truth; upper bound only):",
      round(oracle_dice.mean(), 4))
print("Selector improves over baseline:",
      int((selected_dice > baseline_dice + 1e-9).sum()), "/", len(df))
print("Selector worsens versus baseline:",
      int((selected_dice < baseline_dice - 1e-9).sum()), "/", len(df))
print("Mean selector change:",
      round((selected_dice - baseline_dice).mean(), 4))

print("\n===== 3. HOW OFTEN DOES THE SELECTOR CHOOSE EACH BOX? =====")
names = ["Original", "Expanded", "Contracted"]
for i, name in enumerate(names):
    count = int((selected == i).sum())
    print(f"{name}: {count}/ {len(df)} ({100 * count / len(df):.1f}%)")

print("\n===== 4. STABILITY VS ACTUAL DICE =====")
print("Spearman:",
      round(spearmanr(df["mean_mask_stability"], df["base_dice"]).statistic, 4))
