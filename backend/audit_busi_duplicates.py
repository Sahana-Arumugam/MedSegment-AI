from pathlib import Path
from collections import defaultdict
import csv
import hashlib

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "backend" / "prepared_dataset" / "busi_manifest.csv"
OUTPUT = ROOT / "backend" / "prepared_dataset" / "duplicate_audit.csv"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

with MANIFEST.open("r", newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

# Group identical file contents, not filenames.
hash_groups = defaultdict(list)
missing = []

for row in records:
    path = Path(row["image_path"])
    if not path.is_file():
        missing.append(str(path))
        continue
    digest = sha256_file(path)
    hash_groups[digest].append({
        "image_path": str(path),
        "category": row["category"],
        "split": row["split"],
    })

duplicate_groups = [
    group for group in hash_groups.values() if len(group) > 1
]

cross_split_groups = [
    group for group in duplicate_groups
    if len({item["split"] for item in group}) > 1
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["sha256", "image_path", "category", "split",
                       "group_size", "cross_split_duplicate"]
    )
    writer.writeheader()
    for digest, group in hash_groups.items():
        if len(group) <= 1:
            continue
        cross_split = len({item["split"] for item in group}) > 1
        for item in group:
            writer.writerow({
                "sha256": digest,
                **item,
                "group_size": len(group),
                "cross_split_duplicate": cross_split,
            })

print("BUSI exact-duplicate audit")
print("--------------------------")
print("Manifest records:", len(records))
print("Files missing:", len(missing))
print("Unique image hashes:", len(hash_groups))
print("Exact duplicate groups:", len(duplicate_groups))
print("Duplicate groups crossing splits:", len(cross_split_groups))
print("Audit CSV:", OUTPUT)

if missing:
    print("\nFirst missing files:")
    for p in missing[:10]:
        print(p)

if cross_split_groups:
    print("\nExamples of duplicates across splits:")
    for group in cross_split_groups[:10]:
        print()
        for item in group:
            print(item["split"], "|", item["category"], "|", item["image_path"])
