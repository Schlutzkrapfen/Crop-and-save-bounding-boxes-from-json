"""
Label Studio export -> cropped, sorted training images
==========================================================

Takes the JSON export from Label Studio (bounding boxes) and produces one
cropped image per annotated box, sorted into folders per material class.
The result fits directly into the folder structure expected by
train_crowns.py.

REQUIREMENTS:
- In Label Studio: Export -> choose format "JSON" (not JSON-MIN, this
  script expects the full JSON format with percentage coordinates)
- The original images must be available locally (adjust path below)

USAGE:
    python export_crops.py

Adjust before running:
    LABEL_STUDIO_JSON   -> path to the exported JSON file
    IMAGES_DIR           -> folder with the original (uncropped) images
    OUTPUT_DIR            -> destination folder, will be auto-populated
                             with one subfolder per class
"""

import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

# ---------------------------------------------------------------------------
# Configuration - ADJUST HERE
# ---------------------------------------------------------------------------
LABEL_STUDIO_JSON = "label_studio_export.json"
IMAGES_DIR = "raw_images"                 # folder with the original images
OUTPUT_DIR = "crown_data"                 # destination folder (auto-created)
PADDING_PERCENT = 5                       # extra margin around the box (0 = exact box)

# ---------------------------------------------------------------------------


def find_local_image(image_field: str, images_dir: str) -> str:
    """Resolves the local file path from Label Studio's 'image' field,
    which is often a URL or an upload path."""
    filename = os.path.basename(unquote(urlparse(image_field).path))
    local_path = os.path.join(images_dir, filename)
    if os.path.exists(local_path):
        return local_path
    # Fallback: Label Studio sometimes prefixes filenames like "abc123-image.jpg"
    # -> try to find the original name after the first hyphen
    if "-" in filename:
        stripped = filename.split("-", 1)[1]
        alt_path = os.path.join(images_dir, stripped)
        if os.path.exists(alt_path):
            return alt_path
    raise FileNotFoundError(f"Image not found for: {image_field} (looked for: {local_path})")


def crop_with_padding(image: Image.Image, x_pct, y_pct, w_pct, h_pct, padding_pct):
    img_w, img_h = image.size

    x = x_pct / 100 * img_w
    y = y_pct / 100 * img_h
    w = w_pct / 100 * img_w
    h = h_pct / 100 * img_h

    pad_x = w * (padding_pct / 100)
    pad_y = h * (padding_pct / 100)

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(img_w, x + w + pad_x)
    bottom = min(img_h, y + h + pad_y)

    return image.crop((int(left), int(top), int(right), int(bottom)))


def main():
    with open(LABEL_STUDIO_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    counters = {}
    skipped = 0
    saved = 0

    for task in tasks:
        image_field = task["data"].get("image") or task["data"].get("img")
        try:
            local_image_path = find_local_image(image_field, IMAGES_DIR)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            skipped += 1
            continue

        image = Image.open(local_image_path).convert("RGB")

        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue

                value = result["value"]
                labels = value.get("rectanglelabels", [])
                if not labels:
                    continue
                label = labels[0]

                cropped = crop_with_padding(
                    image,
                    value["x"], value["y"], value["width"], value["height"],
                    PADDING_PERCENT,
                )

                class_dir = Path(OUTPUT_DIR) / label
                class_dir.mkdir(parents=True, exist_ok=True)

                counters[label] = counters.get(label, 0) + 1
                out_name = f"{Path(local_image_path).stem}_{counters[label]:03d}.jpg"
                cropped.save(class_dir / out_name, quality=95)
                saved += 1

    print(f"\nDone. {saved} cropped images saved in '{OUTPUT_DIR}/'.")
    if skipped:
        print(f"{skipped} images skipped (not found, see warnings above).")
    print("\nImages per class:")
    for label, count in sorted(counters.items()):
        print(f"  {label:20s}: {count}")


if __name__ == "__main__":
    main()
