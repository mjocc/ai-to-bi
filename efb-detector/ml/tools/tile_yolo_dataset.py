#!/usr/bin/env python3
import math
import random
import shutil
from pathlib import Path

from PIL import Image

INPUT_DIR = Path("ml/yolo_dataset_base")
OUTPUT_DIR = Path("ml/yolo_dataset_tiles")
TARGET_BOX_SIZE_PX = 64.0
TILE_SIZE = 640
COVERAGE = 10
FLIP_PROB = 0.5
ROTATE_PROB = 0.5
COLOR_JITTER = 0.2
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["grub"]


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


def box_to_yolo(box, tile_x, tile_y, tile_w, tile_h):
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
    return 0, tcx, tcy, tbw, tbh


def jitter_color(img, amount, rng):
    if amount <= 0:
        return img
    scales = [1.0 + rng.uniform(-amount, amount) for _ in range(3)]
    r, g, b = img.split()
    r = r.point(lambda p: max(0, min(255, int(p * scales[0]))))
    g = g.point(lambda p: max(0, min(255, int(p * scales[1]))))
    b = b.point(lambda p: max(0, min(255, int(p * scales[2]))))
    return Image.merge("RGB", (r, g, b))


def augment_box(box, flip_h=False, flip_v=False, rotate_deg=0):
    cls, x, y, w, h = box
    if flip_h:
        x = 1.0 - x
    if flip_v:
        y = 1.0 - y
    if rotate_deg == 90:
        x, y, w, h = y, 1.0 - x, h, w
    elif rotate_deg == 180:
        x, y = 1.0 - x, 1.0 - y
    elif rotate_deg == 270:
        x, y, w, h = 1.0 - y, x, h, w
    return cls, x, y, w, h


def scale_factor_for_boxes(boxes):
    if not boxes:
        return 1.0

    total = 0.0
    count = 0
    for x1, y1, x2, y2 in boxes:
        bw = max(1e-6, x2 - x1)
        bh = max(1e-6, y2 - y1)
        total += (bw + bh) / 2.0
        count += 1

    if count == 0:
        return 1.0

    avg_box_size = total / float(count)
    return TARGET_BOX_SIZE_PX / avg_box_size if avg_box_size > 0 else 1.0


def random_rotate(img, rng):
    if rng.random() > ROTATE_PROB:
        return img, 0
    angle = rng.choice([90, 180, 270])
    return img.rotate(angle), angle


def tiles_for_image(image_w, image_h, tile_size, coverage):
    area = image_w * image_h
    tile_area = tile_size * tile_size
    return max(1, int(math.ceil((area / tile_area) * coverage)))


def random_tile_xy(image_w, image_h, tile_w, tile_h, rng):
    mid_x = rng.uniform(0, image_w)
    mid_y = rng.uniform(0, image_h)
    x0 = int(round(mid_x - tile_w / 2.0))
    y0 = int(round(mid_y - tile_h / 2.0))
    x0 = int(clamp(x0, 0, max(0, image_w - tile_w)))
    y0 = int(clamp(y0, 0, max(0, image_h - tile_h)))
    return x0, y0


def tile_split(in_img, in_lbl, out_img, out_lbl, rng):
    tile_size = max(32, int(TILE_SIZE))
    coverage = COVERAGE

    tile_count = 0
    box_count = 0

    for img_path in find_images(in_img):
        image = Image.open(img_path).convert("RGB")
        image_w, image_h = image.size
        boxes = read_labels(in_lbl / f"{img_path.stem}.txt", image_w, image_h)

        scale = scale_factor_for_boxes(boxes)
        if image_w < tile_size or image_h < tile_size:
            min_scale = max(tile_size / image_w, tile_size / image_h)
            if min_scale > 1.0:
                scale = max(scale, min_scale)

        if scale != 1.0:
            new_w = max(tile_size, int(round(image_w * scale)))
            new_h = max(tile_size, int(round(image_h * scale)))
            image = image.resize((new_w, new_h))
            boxes = [
                (x1 * (new_w / image_w), y1 * (new_h / image_h), x2 * (new_w / image_w), y2 * (new_h / image_h))
                for x1, y1, x2, y2 in boxes
            ]
            image_w, image_h = new_w, new_h

        n_tiles = tiles_for_image(image_w, image_h, tile_size, coverage)
        for i in range(n_tiles):
            x0, y0 = random_tile_xy(image_w, image_h, tile_size, tile_size, rng)

            x1 = min(image_w, x0 + tile_size)
            y1 = min(image_h, y0 + tile_size)
            crop_w = x1 - x0
            crop_h = y1 - y0

            tile = image.crop((x0, y0, x1, y1)).resize((tile_size, tile_size))
            tile = jitter_color(tile, COLOR_JITTER, rng)

            flip_h = rng.random() < FLIP_PROB
            flip_v = rng.random() < FLIP_PROB
            if flip_h:
                tile = tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if flip_v:
                tile = tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            tile, rot = random_rotate(tile, rng)

            labels = []
            for box in boxes:
                label = box_to_yolo(box, x0, y0, crop_w, crop_h)
                if label is None:
                    continue
                label = augment_box(
                    label,
                    flip_h=flip_h,
                    flip_v=flip_v,
                    rotate_deg=rot,
                )
                _, cx, cy, bw, bh = label
                labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            tile_name = f"{img_path.stem}__r{i}__x{x0}_y{y0}{img_path.suffix.lower()}"
            tile.save(out_img / tile_name)
            (out_lbl / f"{Path(tile_name).stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
            )

            tile_count += 1
            box_count += len(labels)

    return tile_count, box_count


def tile_train_split(input_dir, output_dir, rng):
    in_img = input_dir / "images" / "train"
    in_lbl = input_dir / "labels" / "train"
    out_img = output_dir / "images" / "train"
    out_lbl = output_dir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    return tile_split(in_img, in_lbl, out_img, out_lbl, rng)


def tile_val_split(input_dir, output_dir, rng):
    in_img = input_dir / "images" / "val"
    in_lbl = input_dir / "labels" / "val"
    out_img = output_dir / "images" / "val"
    out_lbl = output_dir / "labels" / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    return tile_split(in_img, in_lbl, out_img, out_lbl, rng)


def main():
    if not (INPUT_DIR / "data.yaml").exists():
        raise FileNotFoundError(f"Missing data.yaml in {INPUT_DIR}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    train_tiles, train_boxes = tile_train_split(INPUT_DIR, OUTPUT_DIR, rng)
    val_tiles, val_boxes = tile_val_split(INPUT_DIR, OUTPUT_DIR, rng)

    out_yaml = OUTPUT_DIR / "data.yaml"
    out_yaml.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_DIR.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "nc: 1",
                f"names: {CLASS_NAMES}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"train tiles: {train_tiles}, train boxes: {train_boxes}")
    print(f"val tiles: {val_tiles}, val boxes: {val_boxes}")
    print(f"data.yaml: {out_yaml}")


if __name__ == "__main__":
    main()
