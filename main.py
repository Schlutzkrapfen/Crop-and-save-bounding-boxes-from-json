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
from urllib.parse import unquote, urlparse, parse_qs

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Configuration - ADJUST HERE
# ---------------------------------------------------------------------------
LABEL_STUDIO_JSON = "thooth.json"
IMAGES_DIR = Path("./Images")                 # folder with the original images
OUTPUT_DIR = "output"                 # destination folder (auto-created)
PADDING_PERCENT = 5                       # extra margin around the box (0 = exact box)

# ---------------------------------------------------------------------------



def get_images_names(directory: Path) -> set[Path]:
    """Extracts the base names (without extensions) of all Images files in a directory.

        Filters for files ending in common image extensions (.png, .jpg, .jpeg).

        Args:
            directory: The Path object pointing to the directory to search.

        Returns:
            A set of Path objects for all matching image files.
        """

    test:set[Path] = set()
    for f in directory.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            test.add(Path(f))
    return test


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
        try:
            local_image_paths = get_images_names(Path(IMAGES_DIR))
            print(local_image_paths)

        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            skipped += 1
            continue

        for local_image_path in local_image_paths:
            try:
                image = Image.open(local_image_path).convert("RGB")
            except Image.UnidentifiedImageError:
                print(f"WARNING something wrong with: {local_image_path}")
                skipped += 1
                continue

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
                    try:
                        cropped.save(class_dir / out_name, quality=95)
                        saved += 1
                    except (ValueError, OSError) as e:
                        print(f"WARNING: {e}")
                        skipped += 1


    print(f"\nDone. {saved} cropped images saved in '{OUTPUT_DIR}/'.")
    if skipped:
        print(f"{skipped} images skipped (not found, see warnings above).")
    print("\nImages per class:")
    for label, count in sorted(counters.items()):
        print(f"  {label:20s}: {count}")


if __name__ == "__main__":
    main()
