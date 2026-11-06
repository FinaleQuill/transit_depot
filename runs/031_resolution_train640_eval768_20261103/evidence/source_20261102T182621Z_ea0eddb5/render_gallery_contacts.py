"""Derive eight contact sheets from frozen gallery comparisons only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parent
GALLERY = PROJECT / "artifacts/tight_crop_gallery_20261102"
METRICS = GALLERY / "metrics.json"
INDEX = GALLERY / "index.md"
VIEWS = ("AP", "PA", "Lateral", "Missing")

# preview_tight_crop_gallery.composite: side=420, gap=16, top=106,
# bottom=38; bottom-row positions use x=gap+col*(side+gap), y=580.
PANEL_SIZE = (1324, 1000)
BASELINE_BOX = (452, 580, 872, 1000)
CANDIDATE_BOX = (888, 580, 1308, 1000)


def main() -> None:
    data = json.loads(METRICS.read_text())
    if not data["complete"] or len(data["items"]) != 96:
        raise ValueError("Expected complete 96-image gallery")
    groups = defaultdict(list)
    for item in data["items"]:
        groups[(item["split"], item["view"])].append(item)
    expected = {(split, view) for split in ("train", "valid") for view in VIEWS}
    if set(groups) != expected or any(len(rows) != 12 for rows in groups.values()):
        raise ValueError("Unexpected selected gallery composition")
    names = [f"contact_{split}_{view}.jpg" for split in ("train", "valid") for view in VIEWS]
    if any((GALLERY / name).exists() for name in names):
        raise FileExistsError("Contact sheet already exists")
    if "## Derived contact sheets" in INDEX.read_text():
        raise ValueError("Index already has contact sheet links")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16)
    small = ImageFont.truetype(font_path, 12)
    sheet_width, sheet_height = 3 * 408 + 4 * 16, 82 + 4 * 238 + 16
    for split in ("train", "valid"):
        for view in VIEWS:
            sheet = Image.new("RGB", (sheet_width, sheet_height), (29, 34, 39))
            draw = ImageDraw.Draw(sheet)
            draw.text((16, 11), f"{split} / {view} | selected review examples (12)", font=font, fill="white")
            draw.text((16, 40), "Full-view baseline (left) vs new crop (right); thumbnails derived from stored comparisons only.",
                      font=small, fill="white")
            for index, item in enumerate(groups[(split, view)]):
                image_path = GALLERY / item["comparison"]
                with Image.open(image_path) as comparison:
                    if comparison.size != PANEL_SIZE:
                        raise ValueError(f"Unexpected comparison layout: {image_path}")
                    baseline = comparison.crop(BASELINE_BOX).resize((188, 188), Image.Resampling.LANCZOS)
                    candidate = comparison.crop(CANDIDATE_BOX).resize((188, 188), Image.Resampling.LANCZOS)
                col, row = index % 3, index // 3
                x, y = 16 + col * 424, 82 + row * 238
                sheet.paste(baseline, (x, y + 28))
                sheet.paste(candidate, (x + 204, y + 28))
                status = "CUT" if item["applied"] else "NO CUT"
                label = (f"{item['image_id']}  {status}  "
                         f"cut {item['removed_area_fraction']:.1%}  gain {item['effective_linear_magnification']:.3f}x")
                draw.text((x, y), label, font=small, fill="white")
                draw.text((x, y + 219), "Full-view 768", font=small, fill=(190, 222, 226))
                draw.text((x + 204, y + 219), "New crop 768", font=small, fill=(255, 171, 165))
            sheet.save(GALLERY / f"contact_{split}_{view}.jpg", quality=91, subsampling=0)
    lines = ["", "## Derived contact sheets", "",
             "These are 768 baseline/candidate thumbnails cropped from the already saved six-panel comparisons. "
             "No source image was reopened and no detection was rerun. Counts describe only the selected 96 review images.", ""]
    for split in ("train", "valid"):
        for view in VIEWS:
            lines.append(f"- [{split} / {view} (12)](contact_{split}_{view}.jpg)")
    with INDEX.open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    print("8 contact sheets:", GALLERY)


if __name__ == "__main__":
    main()
