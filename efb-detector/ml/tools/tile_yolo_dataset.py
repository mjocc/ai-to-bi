#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import random
import shutil
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile a YOLO dataset with overlap")
    p.add_argument("--input-dir", required=True, help="YOLO dataset dir containing images/, labels/, data.yaml")
    p.add_argument("--output-dir", required=True, help="Output tiled dataset dir")
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.25, help="0.0 to <1.0")
    p.add_argument("--train-keep-prob", type=float, default=1.0, help="Randomly keep train tiles")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def tile_starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    out = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if out[-1] != last:
        out.append(last)
    return out


def read_labels(path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_to_tile_label(
    box: tuple[float, float, float, float],
    tx0: int,
    ty0: int,
    tw: int,
    th: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if not (tx0 <= cx < tx0 + tw and ty0 <= cy < ty0 + th):
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
    split: str,
    input_dir: Path,
    output_dir: Path,
    tile_size: int,
    overlap: float,
    train_keep_prob: float,
    rng: random.Random,
) -> tuple[int, int]:
    in_img = input_dir / "images" / split
    in_lbl = input_dir / "labels" / split
    out_img = output_dir / "images" / split
    out_lbl = output_dir / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    step = max(1, int(tile_size * (1.0 - overlap)))
    tiles_written = 0
    labels_written = 0

    for img_path in sorted(in_img.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes = read_labels(in_lbl / f"{img_path.stem}.txt", w, h)

        xs = tile_starts(w, tile_size, step)
        ys = tile_starts(h, tile_size, step)

        for y0 in ys:
            for x0 in xs:
                if split == "train" and rng.random() > train_keep_prob:
                    continue

                x1 = min(w, x0 + tile_size)
                y1 = min(h, y0 + tile_size)
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


def read_nc_names(data_yaml: Path) -> tuple[int, list[str]]:
    nc = 1
    names: list[str] = ["grub"]
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("nc:"):
            try:
                nc = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("names:"):
            raw = s.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list) and parsed:
                    names = [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                pass
    return nc, names


def main() -> None:
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

    nc, names = read_nc_names(input_dir / "data.yaml")

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
                f"nc: {nc}",
                f"names: {names}",
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
