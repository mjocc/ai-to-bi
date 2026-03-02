#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiled YOLO inference + app JSON")
    p.add_argument("--model", required=True, help="Path to .pt")
    p.add_argument("--image", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-overlay", required=True)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-merge", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
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


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[tuple[float, float, float, float, float]], thr: float) -> list[tuple[float, float, float, float, float]]:
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep: list[tuple[float, float, float, float, float]] = []
    for b in boxes:
        drop = False
        for k in keep:
            if iou((b[0], b[1], b[2], b[3]), (k[0], k[1], k[2], k[3])) >= thr:
                drop = True
                break
        if not drop:
            keep.append(b)
    return keep


@dataclass
class LarvaCellCandidate:
    x: int
    y: int
    radius: int
    bbox: tuple[int, int, int, int]
    larva_score: float


@dataclass
class LarvaeLocatorOutput:
    model: str
    generated_at_utc: str
    image_path: str
    image_width: int
    image_height: int
    confidence_threshold: float
    count: int
    candidates: list[LarvaCellCandidate]
    disclaimer: str


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    image = Image.open(args.image).convert("RGB")
    w, h = image.size

    overlap = clamp(args.overlap, 0.0, 0.99)
    step = max(1, int(args.tile_size * (1.0 - overlap)))
    xs = tile_starts(w, args.tile_size, step)
    ys = tile_starts(h, args.tile_size, step)

    raw_boxes: list[tuple[float, float, float, float, float]] = []

    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + args.tile_size)
            y1 = min(h, y0 + args.tile_size)
            tile = image.crop((x0, y0, x1, y1))

            results = model.predict(tile, conf=args.conf, verbose=False, device=args.device)
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    score = float(b.conf[0].item())
                    gx1 = clamp(bx1 + x0, 0.0, float(w))
                    gy1 = clamp(by1 + y0, 0.0, float(h))
                    gx2 = clamp(bx2 + x0, 0.0, float(w))
                    gy2 = clamp(by2 + y0, 0.0, float(h))
                    raw_boxes.append((gx1, gy1, gx2, gy2, score))

    merged = nms(raw_boxes, args.iou_merge)

    candidates: list[LarvaCellCandidate] = []
    draw = ImageDraw.Draw(image)

    for x1, y1, x2, y2, score in merged:
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        radius = int(round(max(x2 - x1, y2 - y1) / 2.0))
        bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

        candidates.append(
            LarvaCellCandidate(
                x=cx,
                y=cy,
                radius=max(1, radius),
                bbox=bbox,
                larva_score=round(score, 4),
            )
        )

        draw.rectangle(bbox, outline=(0, 255, 0), width=2)
        draw.text((bbox[0], max(0, bbox[1] - 12)), f"{score:.2f}", fill=(0, 255, 0))

    output = LarvaeLocatorOutput(
        model=Path(args.model).name,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        image_path=str(Path(args.image).resolve()),
        image_width=w,
        image_height=h,
        confidence_threshold=args.conf,
        count=len(candidates),
        candidates=candidates,
        disclaimer="YOLO tiled detection output.",
    )

    Path(args.output_json).write_text(json.dumps(asdict(output), indent=2), encoding="utf-8")
    image.save(args.output_overlay)

    print(f"Detected {output.count} candidates")
    print(f"JSON written to {args.output_json}")
    print(f"Overlay written to {args.output_overlay}")


if __name__ == "__main__":
    main()
