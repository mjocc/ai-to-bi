"""
Quick inference script for the frame detector. Loads the model once and
runs predictions either in batch mode (pass image paths as args) or
interactively via a simple prompt loop.
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

# flag names - order must match what the model outputs
FLAG_NAMES = [
    'is_frame_90_degrees',
    'is_rotation_invalid', 
    'is_too_small',
    'is_too_large',
    'is_out_of_bounds',
    'is_blurred'
]


def load_model(model_dir=None):
    """Load a checkpoint from model_dir. Tries .keras and .h5 first,
    falls back to SavedModel format if neither exists.
    """
    import tensorflow as tf
    
    model_dir = model_dir or CONFIG.get('model_save_path')
    
    # Try common Keras checkpoint filenames first, then SavedModel
    for checkpoint_name in ['checkpoint.keras', 'checkpoint.h5', 'new_model.keras']:
        checkpoint_path = os.path.join(model_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            try:
                from tensorflow import keras
                print(f"Loading model from: {checkpoint_path}")
                model = keras.models.load_model(checkpoint_path)
                return model
            except Exception as e:
                # not fatal, just try the next one
                print(f"Could not load {checkpoint_path}: {e}")
                continue

    # no Keras file worked, fall back to SavedModel
    saved_model_path = os.path.join(model_dir, 'saved_model')
    if os.path.exists(saved_model_path):
        print(f"Loading model from TensorFlow SavedModel: {saved_model_path}")
        model = tf.saved_model.load(saved_model_path)
        # wrap it so callers can use .predict() the same way as a Keras model
        class SavedModelWrapper:
            def __init__(self, sm_model):
                self._model = sm_model
                self._signature = sm_model.signatures['serving_default']
            
            def predict(self, x, verbose=0):
                # convert numpy -> tensor, run the signature, return as tuple
                if isinstance(x, np.ndarray):
                    x = tf.constant(x, dtype=tf.float32)
                result = self._signature(x)
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
    """Open an image, resize it to the model input size and normalise to [0,1].
    Returns the array plus the original PIL image and its dimensions.
    """
    img = Image.open(image_path).convert('RGB')
    original_width, original_height = img.size
    img_resized = img.resize((img_width, img_height))
    img_array = np.array(img_resized) / 255.0
    return img_array, img, original_width, original_height


def preprocess_batch(image_paths, img_width, img_height):
    """Preprocess a list of images into a stacked batch array."""
    images = []
    originals = []
    original_sizes = []
    
    for p in image_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            # if the image fails to load just use a blank placeholder
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
    """Run the model on one image and return a dict with bbox, confidence,
    flags and a simple zoom suggestion.
    """
    
    img_array, original_img, orig_w, orig_h = preprocess_image(image_path, img_width, img_height)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model is expected to return (bbox, has_frame, flags)
    bbox_pred, class_pred, flags_pred = model.predict(img_array, verbose=0)

    # un-normalise the bbox back to pixel coords in the original image
    x_norm, y_norm, w_norm, h_norm = bbox_pred[0]
    x = int(x_norm * orig_w)
    y = int(y_norm * orig_h)
    w = int(w_norm * orig_w)
    h = int(h_norm * orig_h)

    # make sure bbox doesn't go outside the image
    x = max(0, min(x, orig_w - 1))
    y = max(0, min(y, orig_h - 1))
    w = max(1, min(w, orig_w - x))
    h = max(1, min(h, orig_h - y))

    has_frame = bool(class_pred[0][0] > 0.5)
    frame_confidence = float(class_pred[0][0])

    flags = {}
    for i, flag_name in enumerate(FLAG_NAMES):
        flags[flag_name] = bool(flags_pred[0][i] > 0.5)

    # work out how much of the image the frame covers, then suggest a zoom
    frame_area = w * h
    image_area = orig_w * orig_h
    frame_ratio = frame_area / image_area if image_area > 0 else 0

    is_too_small = flags.get('is_too_small', False)
    is_too_large = flags.get('is_too_large', False)

    ideal_ratio = 0.45  # targeting ~45% coverage
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
        zoom_instructions.append("Current zoom is optimal")

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
    """Run predictions on a list of images and return a list of result dicts."""
    img_w = CONFIG.get('img_width', 224)
    img_h = CONFIG.get('img_height', 224)
    
    batch, originals, original_sizes = preprocess_batch(image_paths, img_w, img_h)
    
    # Run model prediction on the batch
    bbox_preds, class_preds, flags_preds = model.predict(batch, verbose=0)

    out = []
    for i, (img_path, orig_size) in enumerate(zip(image_paths, original_sizes)):
        orig_w, orig_h = orig_size

        # un-normalise bbox
        x_norm, y_norm, w_norm, h_norm = bbox_preds[i]
        x = int(x_norm * orig_w)
        y = int(y_norm * orig_h)
        w = int(w_norm * orig_w)
        h = int(h_norm * orig_h)

        # clamp to image bounds
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))

        has_frame = bool(class_preds[i][0] > 0.5)
        frame_confidence = float(class_preds[i][0])

        flags = {}
        for j, flag_name in enumerate(FLAG_NAMES):
            flags[flag_name] = bool(flags_preds[i][j] > 0.5)

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

    # Load the model once so we can reuse it for multiple images
    try:
        model = load_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    effective_batch = args.batch_size if getattr(args, 'batch_size', None) else CONFIG.get('batch_size', 32)

    # batch prediction from CLI args
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
                    # trimmed version without all the nested detail
                    out[p] = {
                        'has_frame': r['has_frame'],
                        'confidence': r['confidence'],
                        'bbox': r['bbox'],
                        'zoom': r['zoom']['recommended_zoom'],
                        'instructions': r['zoom']['instructions']
                    }
        print(json.dumps(out, indent=2))
        return

    # Interactive mode: read paths from stdin and print simple results
    if args.interactive:
        print(f"Interactive mode. Type image paths or 'exit' to quit. Batch size: {effective_batch}")
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

    parser.print_help()


if __name__ == '__main__':
    main()

