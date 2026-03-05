#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["grub"]
CLASS_COUNT = 1


def parse_args():
    p = argparse.ArgumentParser(description="Tile YOLO dataset to 640x640")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--train-keep-prob", type=float, default=1.0, help="Randomly keep train tiles")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def tile_start_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


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


def box_to_tile_label(box, tx0, ty0, tw, th):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    if not (tx0 <= center_x < tx0 + tw and ty0 <= center_y < ty0 + th):
        return None

    nx1 = clamp(x1 - tx0, 0.0, float(tw))
    ny1 = clamp(y1 - ty0, 0.0, float(th))
    nx2 = clamp(x2 - tx0, 0.0, float(tw))
    ny2 = clamp(y2 - ty0, 0.0, float(th))

    bw = nx2 - nx1
    bh = ny2 - ny1
    if bw <= 1.0 or bh <= 1.0:
        return None

    tcx = (nx1 + nx2) / 2.0 / tw
    tcy = (ny1 + ny2) / 2.0 / th
    tbw = bw / tw
    tbh = bh / th
    return tcx, tcy, tbw, tbh


def process_split(
    split,
    input_dir,
    output_dir,
    tile_size,
    overlap,
    train_keep_prob,
    rng,
):
    in_img = input_dir / "images" / split
    in_lbl = input_dir / "labels" / split
    out_img = output_dir / "images" / split
    out_lbl = output_dir / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles_written = 0
    labels_written = 0

    for img_path in sorted(in_img.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = Image.open(img_path).convert("RGB")
        image_w, image_h = img.size
        boxes = read_labels(in_lbl / f"{img_path.stem}.txt", image_w, image_h)

        x_starts = tile_start_positions(image_w, tile_size, stride)
        y_starts = tile_start_positions(image_h, tile_size, stride)

        for y0 in y_starts:
            for x0 in x_starts:
                if split == "train" and rng.random() > train_keep_prob:
                    continue

                x1 = min(image_w, x0 + tile_size)
                y1 = min(image_h, y0 + tile_size)
                tw = x1 - x0
                th = y1 - y0

                tile = img.crop((x0, y0, x1, y1))
                labels = []
                for box in boxes:
                    out = box_to_tile_label(box, x0, y0, tw, th)
                    if out is not None:
                        tcx, tcy, tbw, tbh = out
                        labels.append(f"0 {tcx:.6f} {tcy:.6f} {tbw:.6f} {tbh:.6f}")

                tile_name = f"{img_path.stem}__x{x0}_y{y0}{img_path.suffix.lower()}"
                tile.save(out_img / tile_name)
                (out_lbl / f"{Path(tile_name).stem}.txt").write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                )

                tiles_written += 1
                labels_written += len(labels)

    return tiles_written, labels_written


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not (input_dir / "data.yaml").exists():
        raise FileNotFoundError(f"Missing data.yaml in {input_dir}")

    overlap = clamp(args.overlap, 0.0, 0.99)
    keep_prob = clamp(args.train_keep_prob, 0.0, 1.0)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    t_train, l_train = process_split(
        "train", input_dir, output_dir, args.tile_size, overlap, keep_prob, rng
    )
    t_val, l_val = process_split("val", input_dir, output_dir, args.tile_size, overlap, 1.0, rng)

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

    print(f"train tiles: {t_train}, train boxes: {l_train}")
    print(f"val tiles: {t_val}, val boxes: {l_val}")
    print(f"data.yaml: {out_yaml}")


if __name__ == "__main__":
    main()
