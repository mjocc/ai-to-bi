#!/usr/bin/env python3
import json
import random
import shutil
from pathlib import Path

INPUT_DIR = Path("ml/grub_dataset")
OUTPUT_DIR = Path("ml/yolo_dataset_base")
TRAIN_RATIO = 0.8
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAME = "grub"


def clamp_box(x1, y1, x2, y2, width, height):
    left = max(0.0, min(width, min(x1, x2)))
    right = max(0.0, min(width, max(x1, x2)))
    top = max(0.0, min(height, min(y1, y2)))
    bottom = max(0.0, min(height, max(y1, y2)))
    return left, top, right, bottom


def box_to_yolo(x1, y1, x2, y2, width, height):
    left, top, right, bottom = clamp_box(x1, y1, x2, y2, float(width), float(height))
    box_w = right - left
    box_h = bottom - top
    cx = left + box_w / 2.0
    cy = top + box_h / 2.0
    return cx / width, cy / height, box_w / width, box_h / height


def main():
    images = [
        path
        for path in sorted(INPUT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    rng = random.Random(SEED)
    rng.shuffle(images)
    train_count = int(len(images) * max(0.0, min(1.0, float(TRAIN_RATIO))))
    train_set = set(images[:train_count])

    image_train_dir = OUTPUT_DIR / "images" / "train"
    image_val_dir = OUTPUT_DIR / "images" / "val"
    label_train_dir = OUTPUT_DIR / "labels" / "train"
    label_val_dir = OUTPUT_DIR / "labels" / "val"

    for d in (image_train_dir, image_val_dir, label_train_dir, label_val_dir):
        d.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for image_path in images:
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Skipping {image_path.name}: missing {json_path.name}")
            skipped += 1
            continue

        data = json.loads(json_path.read_text(encoding="utf-8"))
        width = int(data.get("imageWidth", 0))
        height = int(data.get("imageHeight", 0))

        if width <= 0 or height <= 0:
            print(f"Skipping {image_path.name}: invalid imageWidth/imageHeight")
            skipped += 1
            continue

        lines = []

        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            x1, y1 = map(float, points[0])
            x2, y2 = map(float, points[1])

            cx, cy, bw, bh = box_to_yolo(x1, y1, x2, y2, width, height)
            if bw <= 0.0 or bh <= 0.0:
                continue

            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        split = "train" if image_path in train_set else "val"
        dest_image = (image_train_dir if split == "train" else image_val_dir) / image_path.name
        dest_label = (label_train_dir if split == "train" else label_val_dir) / f"{image_path.stem}.txt"

        shutil.copy2(image_path, dest_image)
        dest_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1

    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_DIR.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "nc: 1",
                f"names: ['{CLASS_NAME}']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"YOLO dataset written to: {OUTPUT_DIR}")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
