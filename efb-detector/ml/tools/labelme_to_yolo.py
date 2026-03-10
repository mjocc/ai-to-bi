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
CLASS_ID = 0
CLASS_NAME = "grub"


def corners_to_yolo(x1, y1, x2, y2, width, height):
    left = max(0.0, min(float(width), min(x1, x2)))
    right = max(0.0, min(float(width), max(x1, x2)))
    top = max(0.0, min(float(height), min(y1, y2)))
    bottom = max(0.0, min(float(height), max(y1, y2)))

    box_w = max(0.0, right - left)
    box_h = max(0.0, bottom - top)

    cx = left + box_w / 2.0
    cy = top + box_h / 2.0

    return cx / width, cy / height, box_w / width, box_h / height


def find_images(input_dir):
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main():
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    train_ratio = max(0.0, min(1.0, float(TRAIN_RATIO)))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    images = list(find_images(input_dir))
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    rng = random.Random(SEED)
    rng.shuffle(images)

    train_count = int(len(images) * train_ratio)
    train_set = set(images[:train_count])

    image_train_dir = output_dir / "images" / "train"
    image_val_dir = output_dir / "images" / "val"
    label_train_dir = output_dir / "labels" / "train"
    label_val_dir = output_dir / "labels" / "val"

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

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        width = int(payload.get("imageWidth", 0))
        height = int(payload.get("imageHeight", 0))

        if width <= 0 or height <= 0:
            print(f"Skipping {image_path.name}: invalid imageWidth/imageHeight")
            skipped += 1
            continue

        yolo_lines = []
        shapes = payload.get("shapes", [])

        for shape in shapes:
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            p1, p2 = points[0], points[1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])

            cx, cy, bw, bh = corners_to_yolo(x1, y1, x2, y2, width, height)
            if bw <= 0.0 or bh <= 0.0:
                continue

            yolo_lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        split = "train" if image_path in train_set else "val"
        dest_image = (image_train_dir if split == "train" else image_val_dir) / image_path.name
        dest_label = (label_train_dir if split == "train" else label_val_dir) / f"{image_path.stem}.txt"

        shutil.copy2(image_path, dest_image)
        dest_label.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        converted += 1

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
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
    print(f"YOLO dataset written to: {output_dir}")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
