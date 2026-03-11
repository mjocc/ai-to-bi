#!/usr/bin/env python3
"""
Small webcam demo that mimics the frame detector without TensorFlow.

This is a lightweight heuristic used mainly for integration tests or to
exercise the video capture and drawing code. It looks for the largest contour
in the frame, checks blur via Laplacian variance, and accepts bounding boxes
within configurable size ratios.

Example:
    python3 live_inference_dummy.py --camera 0 --show
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path


def is_blurry(gray, thresh):
    """Return True if the image is likely blurry based on Laplacian variance."""
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thresh


def find_largest_rect(contours):
    """Return the contour with the largest area and that area value."""
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > best_area:
            best_area = area
            best = c
    return best, best_area


def main():
    parser = argparse.ArgumentParser(description='Dummy live inference (webcam)')
    parser.add_argument('--camera', default=0, help='Camera index or video path')
    parser.add_argument('--show', action='store_true', help='Show preview window')
    # Tunable thresholds. Defaults are intentionally lenient to reduce false
    # negatives during simple testing.
    parser.add_argument('--blur-threshold', type=float, default=40.0,
                        help='Variance of Laplacian below which an image is considered blurry (lower = less sensitive)')
    parser.add_argument('--min-ratio', type=float, default=0.1,
                        help='Minimum bbox/image area ratio to accept (smaller = less sensitive to small frames)')
    parser.add_argument('--max-ratio', type=float, default=0.9,
                        help='Maximum bbox/image area ratio to accept (larger = less sensitive to large frames)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.camera) if str(args.camera).isdigit() else args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera/source: {args.camera}")
        return 1

    found = False
    frame_idx = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame read failed")
            break

        frame_idx += 1
        h, w = frame.shape[:2]
        image_area = h * w

        # Convert to grayscale, check blur, and find contours to locate a
        # candidate rectangle.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_flag = is_blurry(gray, args.blur_threshold)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour, best_area = find_largest_rect(contours)

        bbox = None
        size_ratio = 0.0
        rect_quality = 0.0

        # Allow small contours (low threshold) so simple scenes still produce
        # candidates during tests.
        if best_contour is not None and best_area > 50:
            x, y, bw, bh = cv2.boundingRect(best_contour)
            bbox = (x, y, bw, bh)
            size_ratio = (bw * bh) / float(image_area)
            # measure how well contour fills bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [best_contour], -1, 255, -1)
            contour_area = best_area
            bbox_area = bw * bh
            rect_quality = contour_area / bbox_area if bbox_area > 0 else 0.0

            # Decide if this is a "good" frame
            is_good = False
            reasons = []
            if blur_flag:
                reasons.append('blurry')
            if bbox is None:
                reasons.append('no_contour')
            else:
                if size_ratio < args.min_ratio:
                    reasons.append('too_small')
                elif size_ratio > args.max_ratio:
                    reasons.append('too_large')
                # Relax rect quality requirement to be less strict (allow looser contours)
                if rect_quality < 0.4:
                    reasons.append('poor_rect')

            if not reasons:
                is_good = True

            # Draw overlays
            vis = frame.copy()
            if bbox:
                x, y, bw, bh = bbox
                color = (0, 255, 0) if is_good else (0, 0, 255)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(vis, f"ratio={size_ratio:.2f} rectq={rect_quality:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            status = 'GOOD' if is_good else ('BAD: ' + ','.join(reasons))
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            if args.show:
                cv2.imshow('live_test', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('User quit')
                    break

            if is_good:
                # Save the frame and exit
                out_path = Path('best_frame_webcam.jpg')
                cv2.imwrite(str(out_path), frame)
                elapsed = time.time() - start
                print(f"Found good frame at index {frame_idx} after {elapsed:.2f}s; saved to {out_path}")
                found = True
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    if not found:
        print('No suitable frame found')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
