from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "backend" / "prepared_dataset"
INPUT = DATA / "perceptual_duplicate_candidates.csv"
OUTPUT = DATA / "perceptual_review"

OUTPUT.mkdir(parents=True, exist_ok=True)

with INPUT.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

pairs = [
    r for r in rows
    if r["cross_split"].strip().lower() == "true"
    and int(r["phash_distance"]) <= 1
]

pairs.sort(key=lambda r: int(r["phash_distance"]))

print("Cross-split pairs with pHash distance <= 1:", len(pairs))

if not pairs:
    print("No pairs to review.")
    raise SystemExit(0)

# Each pair gets one panel; 15 panels per contact sheet.
per_page = 15
panel_w, panel_h = 640, 245
thumb_w, thumb_h = 300, 190

for page_start in range(0, len(pairs), per_page):
    page_pairs = pairs[page_start:page_start + per_page]
    cols = 3
    rows_count = math.ceil(len(page_pairs) / cols)

    sheet = Image.new(
        "RGB", (cols * panel_w, rows_count * panel_h), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for local_i, pair in enumerate(page_pairs):
        x = (local_i % cols) * panel_w
        y = (local_i // cols) * panel_h

        draw.rectangle(
            [x + 2, y + 2, x + panel_w - 3, y + panel_h - 3],
            outline="gray", width=2
        )

        draw.text(
            (x + 8, y + 6),
            f"Pair {page_start + local_i + 1} | "
            f"distance={pair['phash_distance']}",
            fill="black"
        )

        for side, key, split_key in [
            (0, "image_a", "split_a"),
            (1, "image_b", "split_b"),
        ]:
            path = Path(pair[key])
            tx = x + 8 + side * 315
            ty = y + 28

            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb_w, thumb_h))
                    px = tx + (thumb_w - im.width) // 2
                    sheet.paste(im, (px, ty))
            except Exception as exc:
                draw.text((tx, ty), f"Could not load: {exc}", fill="red")

            draw.text(
                (tx, y + 220),
                f"{pair[split_key]} | {path.name}",
                fill="black"
            )

    page_number = page_start // per_page + 1
    out = OUTPUT / f"contact_sheet_{page_number:02d}.jpg"
    sheet.save(out, quality=92)
    print(out)

print("\nOpen the generated contact sheets and inspect the pairs.")
