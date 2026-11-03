from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "backend" / "prepared_dataset"
INPUT = DATA / "perceptual_duplicate_candidates.csv"
OUTPUT = DATA / "pixel_similarity_review.csv"

with INPUT.open("r", newline="", encoding="utf-8-sig") as f:
    pairs = list(csv.DictReader(f))

pairs = [
    p for p in pairs
    if p["cross_split"].strip().lower() == "true"
    and int(p["phash_distance"]) <= 1
]

results = []

for p in pairs:
    a = cv2.imread(p["image_a"], cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(p["image_b"], cv2.IMREAD_GRAYSCALE)

    if a is None or b is None:
        results.append({
            **p, "same_dimensions": False,
            "decoded_pixels_identical": False,
            "resized_mean_abs_difference": "",
            "resized_exact_pixel_fraction": "",
            "status": "image_read_error",
        })
        continue

    same_dimensions = a.shape == b.shape
    identical = same_dimensions and np.array_equal(a, b)

    a_small = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)
    b_small = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(a_small, b_small)
    mae = float(diff.mean())
    exact_fraction = float(np.mean(a_small == b_small))

    results.append({
        **p,
        "same_dimensions": same_dimensions,
        "decoded_pixels_identical": identical,
        "resized_mean_abs_difference": round(mae, 4),
        "resized_exact_pixel_fraction": round(exact_fraction, 4),
        "status": "exact_decoded_duplicate" if identical else "needs_visual_review",
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    fields = list(results[0].keys()) if results else []
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("Pairs checked:", len(results))
print("Decoded pixel-identical pairs:", sum(
    r["decoded_pixels_identical"] for r in results
))
print("Pairs with matching dimensions:", sum(
    r["same_dimensions"] for r in results
))
print("\nClosest pixel matches:")
for r in sorted(results, key=lambda x: (
    x["resized_mean_abs_difference"] == "",
    float(x["resized_mean_abs_difference"] or 999)
))[:15]:
    print(
        f"pHash={r['phash_distance']} | "
        f"pixel_identical={r['decoded_pixels_identical']} | "
        f"MAE={r['resized_mean_abs_difference']} | "
        f"{Path(r['image_a']).name} [{r['split_a']}] <> "
        f"{Path(r['image_b']).name} [{r['split_b']}]"
    )

print("\nCSV:", OUTPUT)
