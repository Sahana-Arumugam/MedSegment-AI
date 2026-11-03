from pathlib import Path
from collections import defaultdict
from itertools import combinations
import csv

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "backend" / "prepared_dataset" / "busi_manifest.csv"
OUTPUT = ROOT / "backend" / "prepared_dataset" / "perceptual_duplicate_candidates.csv"

with MANIFEST.open("r", newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

items = []
missing = []

for row in records:
    path = Path(row["image_path"])
    if not path.is_file():
        missing.append(str(path))
        continue

    try:
        with Image.open(path) as img:
            phash = imagehash.phash(img.convert("RGB"))
        items.append({
            "path": str(path),
            "category": row["category"],
            "split": row["split"],
            "hash": phash,
        })
    except Exception as exc:
        print(f"Could not process {path}: {exc}")

# Compare every image pair. Hamming distance <= 6 is a
# candidate threshold, not proof that images are duplicates.
candidates = []

for a, b in combinations(items, 2):
    distance = a["hash"] - b["hash"]
    if distance <= 6:
        candidates.append({
            "image_a": a["path"],
            "category_a": a["category"],
            "split_a": a["split"],
            "image_b": b["path"],
            "category_b": b["category"],
            "split_b": b["split"],
            "phash_distance": distance,
            "cross_split": a["split"] != b["split"],
        })

candidates.sort(key=lambda x: x["phash_distance"])

fields = [
    "image_a", "category_a", "split_a",
    "image_b", "category_b", "split_b",
    "phash_distance", "cross_split",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(candidates)

cross_split = [x for x in candidates if x["cross_split"]]

print("\nBUSI perceptual-hash audit")
print("--------------------------")
print("Manifest records:", len(records))
print("Images processed:", len(items))
print("Missing files:", len(missing))
print("Candidate pairs (distance <= 6):", len(candidates))
print("Candidates crossing splits:", len(cross_split))
print("Output CSV:", OUTPUT)

print("\nClosest candidate pairs:")
for item in candidates[:20]:
    print(
        f"distance={item['phash_distance']} | "
        f"cross_split={item['cross_split']}\n"
        f"  A: [{item['split_a']}] {item['image_a']}\n"
        f"  B: [{item['split_b']}] {item['image_b']}"
    )
