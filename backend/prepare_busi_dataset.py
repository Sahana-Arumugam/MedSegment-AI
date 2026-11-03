
from pathlib import Path
import csv
import random

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "BUSI"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "prepared_dataset"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def find_masks(image_path):
    """Find BUSI masks associated with an original image."""
    folder = image_path.parent
    stem = image_path.stem

    candidates = sorted(folder.glob(f"{stem}_mask*.png"))
    return [
        path for path in candidates
        if path.stem == f"{stem}_mask"
        or path.stem.startswith(f"{stem}_mask_")
    ]


def get_union_mask(mask_paths):
    """Combine all annotations for an image into one binary mask."""
    combined = None

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        binary = mask > 0
        if combined is None:
            combined = binary
        else:
            if combined.shape != binary.shape:
                raise ValueError(
                    f"Mask dimensions differ for {mask_path}"
                )
            combined |= binary

    return combined


def get_bbox(mask):
    """Return an exclusive-coordinate bounding box or None."""
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def discover_images():
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    records = []

    for category in ("benign", "malignant", "normal"):
        category_dir = DATASET_ROOT / category

        if not category_dir.is_dir():
            raise FileNotFoundError(
                f"Expected category directory: {category_dir}"
            )

        originals = sorted(
            path for path in category_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in supported
            and "_mask" not in path.stem.lower()
        )

        for image_path in originals:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"WARNING: unreadable image skipped: {image_path}")
                continue

            height, width = image.shape[:2]
            masks = find_masks(image_path)

            # Normal images can have no lesion annotation.
            if category in ("benign", "malignant") and not masks:
                print(f"WARNING: no mask found: {image_path.name}")

            bbox = None
            if masks:
                union_mask = get_union_mask(masks)

                if union_mask.shape != (height, width):
                    raise ValueError(
                        f"Image/mask dimensions differ: {image_path.name}"
                    )

                bbox = get_bbox(union_mask)

            records.append({
                "image_path": str(image_path.resolve()),
                "category": category,
                "width": width,
                "height": height,
                "mask_paths": ";".join(
                    str(path.resolve()) for path in masks
                ),
                "bbox_x1": "" if bbox is None else bbox[0],
                "bbox_y1": "" if bbox is None else bbox[1],
                "bbox_x2": "" if bbox is None else bbox[2],
                "bbox_y2": "" if bbox is None else bbox[3],
                "has_mask": bool(masks),
                "has_bbox": bbox is not None,
                "split": "",
            })

    return records


def stratified_split(records):
    """Split each category reproducibly into train, validation and test."""
    rng = random.Random(SEED)

    for category in ("benign", "malignant", "normal"):
        group = [r for r in records if r["category"] == category]
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        for record in group[:n_train]:
            record["split"] = "train"

        for record in group[n_train:n_train + n_val]:
            record["split"] = "validation"

        for record in group[n_train + n_val:]:
            record["split"] = "test"


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = discover_images()
    if not records:
        raise RuntimeError("No original BUSI images were found.")

    stratified_split(records)

    manifest_path = OUTPUT_DIR / "busi_manifest.csv"
    fieldnames = list(records[0].keys())

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\nBUSI dataset preparation summary")
    print("--------------------------------")
    print(f"Original images found: {len(records)}")
    print(f"Manifest: {manifest_path}")

    for split in ("train", "validation", "test"):
        subset = [r for r in records if r["split"] == split]
        print(f"\n{split.title()}: {len(subset)}")

        for category in ("benign", "malignant", "normal"):
            count = sum(r["category"] == category for r in subset)
            print(f"  {category}: {count}")

    print("\nAnnotation summary")
    for category in ("benign", "malignant", "normal"):
        subset = [r for r in records if r["category"] == category]
        with_masks = sum(r["has_mask"] for r in subset)
        with_boxes = sum(r["has_bbox"] for r in subset)
        print(
            f"  {category}: {len(subset)} images, "
            f"{with_masks} with masks, {with_boxes} with boxes"
        )

    print(
        "\nNote: these are image-level splits. Patient-level leakage "
        "cannot be ruled out without patient identifiers."
    )


if __name__ == "__main__":
    main()
