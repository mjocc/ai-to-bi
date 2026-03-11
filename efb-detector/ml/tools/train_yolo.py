#!/usr/bin/env python3

from pathlib import Path

from ultralytics import YOLO

DATA_PATH = Path("ml/yolo_dataset_tiles/data.yaml")
WEIGHTS = "yolo11n.pt"
EPOCHS = 200
IMGSZ = 640
BATCH = 4
DEVICE = "mps"
PROJECT = "runs"
NAME = "grub-yolo-v5"
WORKERS = 4
PATIENCE = 30


def main():
    model = YOLO(WEIGHTS)
    model.train(
        data=str(DATA_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        workers=WORKERS,
        patience=PATIENCE,
        pretrained=True,
    )


if __name__ == "__main__":
    main()
