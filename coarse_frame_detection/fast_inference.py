"""
Fast inference utility for Object Detection

Loads the trained Keras model once and allows quick predictions either via
command-line (single or multiple images) or interactively from a REPL loop.

Usage:
  # single/batch prediction
  python fast_inference.py --image tests/notframe1.jpeg
  python fast_inference.py --image img1.jpg img2.jpg img3.jpg

  # interactive REPL (keeps model loaded in memory)
  python fast_inference.py --interactive

This intentionally avoids plotting and other blocking behavior so predictions
are fast. No extra packages required (reuses TensorFlow already in the venv).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np

try:
    from config import TRAINING_CONFIG
    CONFIG = TRAINING_CONFIG
except Exception:
    CONFIG = {
        'img_height': 224,
        'img_width': 224,
        'model_save_path': './models/frame_detector',
        'batch_size': 32,
    }

# Flag names for detection
FLAG_NAMES = [
    'is_frame_90_degrees',
    'is_rotation_invalid', 
    'is_too_small',
    'is_too_large',
    'is_out_of_bounds',
    'is_blurred'
]


def load_model(model_dir=None):
    """Load Keras model from checkpoint file inside model_dir"""
    import tensorflow as tf
    
    model_dir = model_dir or CONFIG.get('model_save_path')
    
    # First try loading as Keras model, then fall back to TensorFlow SavedModel
    for checkpoint_name in ['checkpoint.keras', 'checkpoint.h5', 'new_model.keras']:
        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            try:
                from tensorflow import keras
                print(f"Loading model from: {checkpoint_path}")
                model = keras.models.load_model(checkpoint_path)
                return model
            except Exception as e:
                print(f"Could not load {checkpoint_path}: {e}")
                continue
    
    # Try loading from saved_model format (TensorFlow SavedModel)
    saved_model_path = os.path.join(model_dir, 'saved_model')
    if os.path.exists(saved_model_path):
        print(f"Loading model from TensorFlow SavedModel: {saved_model_path}")
        model = tf.saved_model.load(saved_model_path)
        # Return a wrapper object that provides predict functionality
        class SavedModelWrapper:
            def __init__(self, sm_model):
                self._model = sm_model
                self._signature = sm_model.signatures['serving_default']
            
            def predict(self, x, verbose=0):
                # Convert numpy array to tensor if needed
                if isinstance(x, np.ndarray):
                    x = tf.constant(x, dtype=tf.float32)
                result = self._signature(x)
                # Convert to the same format as Keras model output
                return (
                    result['bbox'].numpy(),
                    result['has_frame'].numpy(),
                    result['flags'].numpy()
                )
            
            def make_predict_function(self):
                pass  # Not needed for SavedModel
        
        return SavedModelWrapper(model)
    
    raise FileNotFoundError(f"No model checkpoint found in: {model_dir}")


def preprocess_image(image_path, img_width, img_height):
    """Preprocess a single image for detection"""
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size
    img_resized = img.resize((img_width, img_height))
    img_array = np.array(img_resized) / 255.0
    return img_array, img, original_width, original_height


def preprocess_batch(image_paths, img_width, img_height):
    """Preprocess multiple images for batch detection"""
    images = []
    originals = []
    original_sizes = []
    
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            # fallback: blank image if load fails
            img = Image.new('RGB', (img_width, img_height))
        
        original_width, original_height = img.size
        img_resized = img.resize((img_width, img_height))
        img_array = np.array(img_resized) / 255.0
        
        images.append(img_array)
        originals.append(img)
        original_sizes.append((original_width, original_height))
    
    batch = np.stack(images, axis=0)
    return batch, originals, original_sizes


def predict_single(model, image_path, img_width, img_height):
    """Predict on a single image and return detection results with zoom info"""
    
    img_array, original_img, orig_w, orig_h = preprocess_image(image_path, img_width, img_height)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict (model outputs: bbox, has_frame, flags)
    bbox_pred, class_pred, flags_pred = model.predict(img_array, verbose=0)
    
    # Denormalize bbox to original image coordinates
    x_norm, y_norm, w_norm, h_norm = bbox_pred[0]
    x = int(x_norm * orig_w)
    y = int(y_norm * orig_h)
    w = int(w_norm * orig_w)
    h = int(h_norm * orig_h)
    
    # Ensure bbox is within image bounds
    x = max(0, min(x, orig_w - 1))
    y = max(0, min(y, orig_h - 1))
    w = max(1, min(w, orig_w - x))
    h = max(1, min(h, orig_h - y))
    
    # Parse classification results
    has_frame = bool(class_pred[0][0] > 0.5)
    frame_confidence = float(class_pred[0][0])
    
    # Parse flags
    flags = {}
    for i, flag_name in enumerate(FLAG_NAMES):
        flags[flag_name] = bool(flags_pred[0][i] > 0.5)
    
    # Calculate zoom recommendation
    frame_area = w * h
    image_area = orig_w * orig_h
    frame_ratio = frame_area / image_area if image_area > 0 else 0
    
    # Determine if frame needs zoom in/out
    is_too_small = flags.get('is_too_small', False)
    is_too_large = flags.get('is_too_large', False)
    
    # Calculate recommended zoom (based on ideal frame coverage ~30-60% of image)
    ideal_ratio = 0.45  # 45% of image
    if frame_ratio > 0:
        recommended_zoom = ideal_ratio / frame_ratio
        recommended_zoom = max(0.5, min(3.0, recommended_zoom))  # Clamp between 0.5x and 3x
    else:
        recommended_zoom = 1.0
    
    # Build zoom instructions
    zoom_instructions = []
    if is_too_small or frame_ratio < 0.2:
        zoom_instructions.append("Move closer to the frame")
        zoom_instructions.append(f"Recommended zoom: {recommended_zoom:.1f}x")
    elif is_too_large or frame_ratio > 0.7:
        zoom_instructions.append("Move back from the frame")
        zoom_instructions.append(f"Recommended zoom: {recommended_zoom:.1f}x")
    else:
        zoom_instructions.append("Frame is at good distance")
        zoom_instructions.append("Current zoom is optimal")
    
    # Add flag-based instructions
    if flags.get('is_blurred', False):
        zoom_instructions.append("Image is blurry - hold camera steady")
    if flags.get('is_out_of_bounds', False):
        zoom_instructions.append("Frame extends beyond image bounds - reposition")
    if flags.get('is_rotation_invalid', False):
        zoom_instructions.append("Frame rotation is invalid - align frame properly")
    if flags.get('is_frame_90_degrees', False):
        zoom_instructions.append("Frame is rotated 90 degrees")
    
    return {
        'has_frame': has_frame,
        'confidence': frame_confidence,
        'bbox': {
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'normalized': {
                'x': float(x_norm),
                'y': float(y_norm),
                'width': float(w_norm),
                'height': float(h_norm)
            }
        },
        'flags': flags,
        'zoom': {
            'frame_ratio': float(frame_ratio),
            'recommended_zoom': float(recommended_zoom),
            'is_optimal': 0.2 <= frame_ratio <= 0.7,
            'needs_zoom_in': is_too_small or frame_ratio < 0.2,
            'needs_zoom_out': is_too_large or frame_ratio > 0.7,
            'instructions': zoom_instructions
        },
        'original_size': {'width': orig_w, 'height': orig_h}
    }


def predict_batch(model, image_paths):
    """Predict on a batch of images"""
    img_w = CONFIG.get('img_width', 224)
    img_h = CONFIG.get('img_height', 224)
    
    batch, originals, original_sizes = preprocess_batch(image_paths, img_w, img_h)
    
    # Predict
    bbox_preds, class_preds, flags_preds = model.predict(batch, verbose=0)
    
    out = []
    for i, (img_path, orig_size) in enumerate(zip(image_paths, original_sizes)):
        orig_w, orig_h = orig_size
        
        # Denormalize bbox
        x_norm, y_norm, w_norm, h_norm = bbox_preds[i]
        x = int(x_norm * orig_w)
        y = int(y_norm * orig_h)
        w = int(w_norm * orig_w)
        h = int(h_norm * orig_h)
        
        # Ensure bounds
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))
        
        # Classification
        has_frame = bool(class_preds[i][0] > 0.5)
        frame_confidence = float(class_preds[i][0])
        
        # Flags
        flags = {}
        for j, flag_name in enumerate(FLAG_NAMES):
            flags[flag_name] = bool(flags_preds[i][j] > 0.5)
        
        # Zoom calculation
        frame_area = w * h
        image_area = orig_w * orig_h
        frame_ratio = frame_area / image_area if image_area > 0 else 0
        
        is_too_small = flags.get('is_too_small', False)
        is_too_large = flags.get('is_too_large', False)
        
        ideal_ratio = 0.45
        if frame_ratio > 0:
            recommended_zoom = ideal_ratio / frame_ratio
            recommended_zoom = max(0.5, min(3.0, recommended_zoom))
        else:
            recommended_zoom = 1.0
        
        zoom_instructions = []
        if is_too_small or frame_ratio < 0.2:
            zoom_instructions.append("Move closer to the frame")
            zoom_instructions.append(f"Recommended zoom: {recommended_zoom:.1f}x")
        elif is_too_large or frame_ratio > 0.7:
            zoom_instructions.append("Move back from the frame")
            zoom_instructions.append(f"Recommended zoom: {recommended_zoom:.1f}x")
        else:
            zoom_instructions.append("Frame is at good distance")
        
        if flags.get('is_blurred', False):
            zoom_instructions.append("Image is blurry - hold camera steady")
        if flags.get('is_out_of_bounds', False):
            zoom_instructions.append("Frame extends beyond image bounds")
        
        result = {
            'has_frame': has_frame,
            'confidence': frame_confidence,
            'bbox': {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'normalized': {
                    'x': float(x_norm),
                    'y': float(y_norm),
                    'width': float(w_norm),
                    'height': float(h_norm)
                }
            },
            'flags': flags,
            'zoom': {
                'frame_ratio': float(frame_ratio),
                'recommended_zoom': float(recommended_zoom),
                'is_optimal': 0.2 <= frame_ratio <= 0.7,
                'needs_zoom_in': is_too_small or frame_ratio < 0.2,
                'needs_zoom_out': is_too_large or frame_ratio > 0.7,
                'instructions': zoom_instructions
            },
            'original_size': {'width': orig_w, 'height': orig_h}
        }
        
        out.append(result)
    
    return out


def main():
    parser = argparse.ArgumentParser(description='Fast inference for frame detector (Object Detection)')
    parser.add_argument('--image', '-i', nargs='+', help='Path(s) to image(s)')
    parser.add_argument('--interactive', action='store_true', help='Start interactive REPL')
    parser.add_argument('--model-dir', help='Override model directory')
    parser.add_argument('--batch-size', type=int, help='Batch size to use for predictions')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    args = parser.parse_args()

    # Load model once
    try:
        model = load_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Determine effective batch size
    effective_batch = args.batch_size if getattr(args, 'batch_size', None) else CONFIG.get('batch_size', 32)

    # One-shot batch prediction (chunk into batches of effective_batch)
    if args.image:
        paths = [str(p) for p in args.image]
        out = {}
        for i in range(0, len(paths), effective_batch):
            batch_paths = paths[i:i+effective_batch]
            results = predict_batch(model, batch_paths)
            for p, r in zip(batch_paths, results):
                if args.verbose:
                    out[p] = r
                else:
                    # Simplified output for non-verbose mode
                    out[p] = {
                        'has_frame': r['has_frame'],
                        'confidence': r['confidence'],
                        'bbox': r['bbox'],
                        'zoom': r['zoom']['recommended_zoom'],
                        'instructions': r['zoom']['instructions']
                    }
        print(json.dumps(out, indent=2))
        return

    # Interactive REPL: read image paths from stdin
    if args.interactive:
        print("Fast inference REPL (Object Detection). Type image paths (one or multiple) or 'exit' to quit.")
        print(f"Using batch size: {effective_batch}")
        print("\nModel outputs:")
        print("  - Bounding box coordinates")
        print("  - Frame presence classification")
        print("  - Quality flags (blurry, too small, too large, etc.)")
        print("  - Zoom recommendations")
        print()
        
        while True:
            try:
                line = input('path> ').strip()
            except EOFError:
                break
            if not line:
                continue
            low = line.lower()
            if low in ('exit', 'quit'):
                break

            parts = line.split()
            # Process in chunks of effective_batch
            for i in range(0, len(parts), effective_batch):
                batch_paths = parts[i:i+effective_batch]
                results = predict_batch(model, batch_paths)
                for p, r in zip(batch_paths, results):
                    status = "FRAME" if r['has_frame'] else "NO FRAME"
                    print(f"\n{p}:")
                    print(f"  {status} (confidence: {r['confidence']:.2%})")
                    print(f"  BBox: x={r['bbox']['x']}, y={r['bbox']['y']}, w={r['bbox']['width']}, h={r['bbox']['height']}")
                    print(f"  Zoom: {r['zoom']['recommended_zoom']:.1f}x (frame covers {r['zoom']['frame_ratio']:.1%} of image)")
                    if r['zoom']['instructions']:
                        print("  Instructions:")
                        for instr in r['zoom']['instructions']:
                            print(f"    - {instr}")
        return

    # If no args provided, show help
    parser.print_help()


if __name__ == '__main__':
    main()

