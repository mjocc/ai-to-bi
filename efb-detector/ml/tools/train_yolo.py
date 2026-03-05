#!/usr/bin/env python3

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on tiled larvae dataset")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda:0, etc")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="grub-detector")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def main():
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
