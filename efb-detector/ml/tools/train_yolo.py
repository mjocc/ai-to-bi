#!/usr/bin/env python3
"""Train YOLO model for larvae detection."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on larvae dataset")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Starting checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda:0, ...")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="grub-detector")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
