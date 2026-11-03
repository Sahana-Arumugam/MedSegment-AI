from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "backend" / "prompt_stability_validation"
SOURCE = RESULTS / "validation_prompt_stability.csv"
OUT = RESULTS / "geometry_analysis.csv"

df = pd.read_csv(SOURCE)

# This first stage checks available evidence before rerunning inference.
print("Existing validation records:", len(df))
print("Existing columns:")
for col in df.columns:
    print(" -", col)

print("\nGeometry analysis plan:")
print(" - Mask area / image area")
print(" - Mask area / prompt box area")
print(" - Connected component count")
print(" - Mask contact with prompt-box boundary")

print("\nImportant:")
print("The existing CSV does not contain predicted masks or their geometry.")
print("Therefore these features cannot be calculated from this CSV alone.")
print("No model inference was run, and no existing results were modified.")
print("\nSource file:", SOURCE)
print("Planned separate output:", OUT)
