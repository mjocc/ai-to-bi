#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def yolo_from_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    left = clamp(min(x1, x2), 0.0, float(width))
    right = clamp(max(x1, x2), 0.0, float(width))
    top = clamp(min(y1, y2), 0.0, float(height))
    bottom = clamp(max(y1, y2), 0.0, float(height))

    box_w = max(0.0, right - left)
    box_h = max(0.0, bottom - top)

    cx = left + box_w / 2.0
    cy = top + box_h / 2.0

    return cx / width, cy / height, box_w / width, box_h / height


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe rectangles to YOLO labels")
    parser.add_argument("--input-dir", required=True, help="Directory containing images + LabelMe json")
    parser.add_argument("--output-dir", required=True, help="Directory to write YOLO dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio [0.0-1.0]")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_ratio = clamp(float(args.train_ratio), 0.0, 1.0)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(images)

    train_count = int(len(images) * train_ratio)
    train_set = set(images[:train_count])

    image_train_dir = output_dir / "images" / "train"
    image_val_dir = output_dir / "images" / "val"
    label_train_dir = output_dir / "labels" / "train"
    label_val_dir = output_dir / "labels" / "val"

    for d in [image_train_dir, image_val_dir, label_train_dir, label_val_dir]:
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

        yolo_lines: list[str] = []
        shapes = payload.get("shapes", [])

        for shape in shapes:
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            p1, p2 = points[0], points[1]
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])

            cx, cy, bw, bh = yolo_from_points(x1, y1, x2, y2, width, height)
            if bw <= 0.0 or bh <= 0.0:
                continue

            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

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
                "names: ['grub']",
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
