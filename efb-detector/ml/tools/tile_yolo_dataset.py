#!/usr/bin/env python3
import argparse
import math
import random
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["grub"]
CLASS_COUNT = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--coverage", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def find_images(path):
    for p in sorted(path.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def read_labels(path, image_w, image_h):
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        x1 = (cx - bw / 2.0) * image_w
        y1 = (cy - bh / 2.0) * image_h
        x2 = (cx + bw / 2.0) * image_w
        y2 = (cy + bh / 2.0) * image_h
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_to_tile_label(box, tile_x, tile_y, tile_w, tile_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if not (tile_x <= cx < tile_x + tile_w and tile_y <= cy < tile_y + tile_h):
        return None

    nx1 = clamp(x1 - tile_x, 0.0, float(tile_w))
    ny1 = clamp(y1 - tile_y, 0.0, float(tile_h))
    nx2 = clamp(x2 - tile_x, 0.0, float(tile_w))
    ny2 = clamp(y2 - tile_y, 0.0, float(tile_h))

    bw = nx2 - nx1
    bh = ny2 - ny1
    if bw <= 1.0 or bh <= 1.0:
        return None

    tcx = (nx1 + nx2) / 2.0 / tile_w
    tcy = (ny1 + ny2) / 2.0 / tile_h
    tbw = bw / tile_w
    tbh = bh / tile_h
    return tcx, tcy, tbw, tbh


def tiles_for_image(image_w, image_h, tile_size, coverage):
    area = image_w * image_h
    tile_area = tile_size * tile_size
    return max(1, int(math.ceil((area / tile_area) * coverage)))


def random_tile_xy(image_w, image_h, tile_size, rng):
    mid_x = rng.uniform(0, image_w)
    mid_y = rng.uniform(0, image_h)
    x0 = int(round(mid_x - tile_size / 2.0))
    y0 = int(round(mid_y - tile_size / 2.0))
    x0 = int(clamp(x0, 0, max(0, image_w - tile_size)))
    y0 = int(clamp(y0, 0, max(0, image_h - tile_size)))
    return x0, y0


def tile_train_split(input_dir, output_dir, tile_size, coverage, rng):
    in_img = input_dir / "images" / "train"
    in_lbl = input_dir / "labels" / "train"
    out_img = output_dir / "images" / "train"
    out_lbl = output_dir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    tile_count = 0
    box_count = 0

    for img_path in find_images(in_img):
        image = Image.open(img_path).convert("RGB")
        image_w, image_h = image.size
        boxes = read_labels(in_lbl / f"{img_path.stem}.txt", image_w, image_h)

        n_tiles = tiles_for_image(image_w, image_h, tile_size, coverage)
        for i in range(n_tiles):
            x0, y0 = random_tile_xy(image_w, image_h, tile_size, rng)
            x1 = min(image_w, x0 + tile_size)
            y1 = min(image_h, y0 + tile_size)
            tile_w = x1 - x0
            tile_h = y1 - y0

            tile = image.crop((x0, y0, x1, y1))
            labels = []
            for box in boxes:
                out = box_to_tile_label(box, x0, y0, tile_w, tile_h)
                if out is None:
                    continue
                tcx, tcy, tbw, tbh = out
                labels.append(f"0 {tcx:.6f} {tcy:.6f} {tbw:.6f} {tbh:.6f}")

            tile_name = f"{img_path.stem}__r{i}__x{x0}_y{y0}{img_path.suffix.lower()}"
            tile.save(out_img / tile_name)
            (out_lbl / f"{Path(tile_name).stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )

            tile_count += 1
            box_count += len(labels)

    return tile_count, box_count


def copy_val_split(input_dir, output_dir):
    in_img = input_dir / "images" / "val"
    in_lbl = input_dir / "labels" / "val"
    out_img = output_dir / "images" / "val"
    out_lbl = output_dir / "labels" / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0

    for img_path in find_images(in_img):
        shutil.copy2(img_path, out_img / img_path.name)
        image_count += 1

    for lbl_path in sorted(in_lbl.glob("*.txt")):
        shutil.copy2(lbl_path, out_lbl / lbl_path.name)
        label_count += 1

    return image_count, label_count


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not (input_dir / "data.yaml").exists():
        raise FileNotFoundError(f"Missing data.yaml in {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_size = max(32, int(args.tile_size))
    coverage = max(0.1, float(args.coverage))
    rng = random.Random(args.seed)

    train_tiles, train_boxes = tile_train_split(input_dir, output_dir, tile_size, coverage, rng)
    val_images, val_labels = copy_val_split(input_dir, output_dir)

    out_yaml = output_dir / "data.yaml"
    out_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                f"nc: {CLASS_COUNT}",
                f"names: {CLASS_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"train tiles: {train_tiles}, train boxes: {train_boxes}")
    print(f"val full images: {val_images}, val labels: {val_labels}")
    print(f"data.yaml: {out_yaml}")


if __name__ == "__main__":
    main()
