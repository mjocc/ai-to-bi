#!/usr/bin/env python3
"""
Run the frame detector over a video file and optionally save an annotated
video or JSON results.

Examples:
    python video_inference.py input.mp4 output.mp4
    python video_inference.py input.mp4 --json output.json
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
import sys

# Add current dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fast_inference import load_model, preprocess_image, CONFIG


def process_video(video_path, output_path=None, json_path=None, sample_every=1):
    """Scan a video, run detection on sampled frames and collect results.

    If `output_path` is provided an annotated video will be written. If
    `json_path` is given the results will be saved as JSON.
    """

    print(f"Loading model...")
    model = load_model()
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read basic video properties for later coordinate conversions
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Iterate over frames, running detection on every `sample_every` frame
    results = []
    frame_idx = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_every == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare a scaled array compatible with the model input
            img_h, img_w = CONFIG.get('img_height', 224), CONFIG.get('img_width', 224)
            frame_resized = cv2.resize(frame_rgb, (img_w, img_h))
            frame_array = frame_resized.astype(np.float32) / 255.0
            frame_array = np.expand_dims(frame_array, axis=0)

            # Run the model
            bbox_pred, class_pred, flags_pred = model.predict(frame_array, verbose=0)
            
            # Extract results
            has_frame = bool(class_pred[0][0] > 0.5)
            confidence = float(class_pred[0][0])
            
            x_norm, y_norm, w_norm, h_norm = bbox_pred[0]
            
            # Convert normalized bbox coordinates back to pixels in the
            # original video frame.
            x = int(x_norm * width)
            y = int(y_norm * height)
            w = int(w_norm * width)
            h = int(h_norm * height)
            
            result = {
                'frame': frame_idx,
                'has_frame': has_frame,
                'confidence': confidence,
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'flags': {
                    'is_blurred': bool(flags_pred[0][0] > 0.5),
                    'is_too_small': bool(flags_pred[0][1] > 0.5),
                    'is_too_large': bool(flags_pred[0][2] > 0.5),
                }
            }
            results.append(result)
            processed += 1
            
            if processed % 30 == 0:
                print(f"  Processed {processed} frames...")
        
        frame_idx += 1
    
    cap.release()
    print(f"Done! Processed {processed} frames out of {total_frames}")
    
    # Optionally write results as JSON
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")
    
    # Optionally create an annotated output video
    if output_path:
        print(f"Creating annotated video: {output_path}")
        
        # Re-open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        result_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_every == 0 and result_idx < len(results):
                r = results[result_idx]
                
                # Draw bounding box
                if r['has_frame']:
                    color = (0, 255, 0)  # Green for frame
                    cv2.rectangle(frame, (r['bbox']['x'], r['bbox']['y']), 
                                  (r['bbox']['x'] + r['bbox']['width'], 
                                   r['bbox']['y'] + r['bbox']['height']), 
                                  color, 2)
                    
                    # Add label
                    label = f"FRAME {r['confidence']:.0%}"
                    cv2.putText(frame, label, (r['bbox']['x'], r['bbox']['y'] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    color = (0, 0, 255)  # Red for no frame
                    label = f"NO FRAME {r['confidence']:.0%}"
                    cv2.putText(frame, label, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                result_idx += 1
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        print(f"Annotated video saved to: {output_path}")
    
    # Print a short summary of the detection pass
    frames_with_frame = sum(1 for r in results if r['has_frame'])
    print(f"\nSummary: {frames_with_frame}/{len(results)} frames had frames detected")
    
    # Find the best frame (first frame with frame detected and no quality issues)
    best_frame = None
    for r in results:
        if r['has_frame']:
            flags = r.get('flags', {})
            # First non-blurred frame
            if not flags.get('is_blurred', False):
                best_frame = r
                break
    
    if best_frame:
        print(f"Best frame: {best_frame['frame']} (confidence: {best_frame['confidence']:.1%})")
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame['frame'])
        ret, frame = cap.read()
        cap.release()
        if ret:
            video_stem = Path(video_path).stem
            best_frame_path = Path(video_path).parent / f"best_frame_{video_stem}.jpg"
            cv2.imwrite(str(best_frame_path), frame)
            print(f"Best frame saved to: {best_frame_path}")
    else:
        print("No suitable frame found")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run frame detection on video')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', nargs='?', help='Output video file (optional)')
    parser.add_argument('--json', help='Save results to JSON file')
    parser.add_argument('--sample', type=int, default=1, help='Process every N frames (default: 1)')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.json, args.sample)


if __name__ == '__main__':
    main()

