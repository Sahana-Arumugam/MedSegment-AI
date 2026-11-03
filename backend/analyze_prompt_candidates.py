import pandas as pd

path = "backend/prompt_stability_validation/validation_prompt_stability.csv"
df = pd.read_csv(path)

scores = df[["base_dice", "expanded_dice", "contracted_dice"]]
names = ["Original", "Expanded 10%", "Contracted 10%"]

print("\n===== WHICH PROMPT WINS? =====")
winner = scores.values.argmax(axis=1)

for i, name in enumerate(names):
    count = int((winner == i).sum())
    print(f"{name}: {count}/ {len(df)} ({count / len(df) * 100:.1f}%)")

print("\n===== ORACLE UPPER BOUND (NOT DEPLOYABLE) =====")
oracle = scores.max(axis=1)
baseline = df["base_dice"]

print("Oracle mean Dice:", round(oracle.mean(), 4))
print("Baseline mean Dice:", round(baseline.mean(), 4))
print("Oracle improvement:", round((oracle - baseline).mean(), 4))
print("Images with any candidate better than baseline:",
      int((oracle > baseline + 1e-9).sum()))

print("\n===== CANDIDATE SPREAD =====")
spread = scores.max(axis=1) - scores.min(axis=1)
print("Mean best-to-worst Dice spread:", round(spread.mean(), 4))
print("Median best-to-worst Dice spread:", round(spread.median(), 4))
print("Images with spread >= 0.05:", int((spread >= 0.05).sum()))
print("Images with spread >= 0.10:", int((spread >= 0.10).sum()))
