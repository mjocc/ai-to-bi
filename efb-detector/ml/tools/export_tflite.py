#!/usr/bin/env python3

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    YOLO(args.model).export(format="tflite", imgsz=args.imgsz, half=args.half, int8=args.int8)


if __name__ == "__main__":
    main()
