"""
Utilities for running the frame detector on images for quick checks and
visualizations.

Use this to inspect model outputs, compute simple accuracy over a split, or
save annotated predictions for debugging.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# When False, visualizations are written to disk but not shown on-screen.
SHOW_PLOTS = True

# Import configuration
try:
    from config import TRAINING_CONFIG
    CONFIG = TRAINING_CONFIG
except ImportError:
    CONFIG = {
        'img_height': 224,
        'img_width': 224,
        'model_save_path': './models/frame_detector',
        'random_seed': 42,
        'batch_size': 32,
    }

# Names of the quality flags produced by the model (order matters)
FLAG_NAMES = [
    'is_frame_90_degrees',
    'is_rotation_invalid',
    'is_too_small',
    'is_too_large',
    'is_out_of_bounds',
    'is_blurred'
]


class FrameDetectorTester:
    """Helper class to run the trained detector and visualize/results."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or CONFIG['model_save_path']
        self.img_height = CONFIG['img_height']
        self.img_width = CONFIG['img_width']
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model from the model directory.

        Exits if the expected checkpoint cannot be found.
        """
        checkpoint_path = os.path.join(self.model_path, 'checkpoint.h5')
        
        if os.path.exists(checkpoint_path):
            print(f"✓ Loading model from: {checkpoint_path}")
            self.model = keras.models.load_model(checkpoint_path)
            print("✓ Model loaded successfully!")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output names: {self.model.output_names}")
        else:
            print(f"✗ Model not found at: {checkpoint_path}")
            print("Please run train_frame_detector.py first")
            sys.exit(1)
    
    def preprocess_image(self, image_path):
        """Load, resize and normalize an image for model input.

        Returns a batch-shaped array along with the original PIL image and
        its original width and height.
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Store original size
        original_width, original_height = img.size
        
        # Resize to model input size
        img_resized = img.resize((self.img_width, self.img_height))
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img, original_width, original_height

    def predict(self, image_path):
        """Run the model on one image and return a detailed result dict.

        The returned dict mirrors the structure used elsewhere in the repo.
        """
        
        # Preprocess
        img_array, original_img, orig_w, orig_h = self.preprocess_image(image_path)
        
        # Predict (model outputs: bbox, has_frame, flags)
        bbox_pred, class_pred, flags_pred = self.model.predict(img_array, verbose=0)

        # Convert the normalized bbox back to pixel coordinates
        x_norm, y_norm, w_norm, h_norm = bbox_pred[0]
        x = int(x_norm * orig_w)
        y = int(y_norm * orig_h)
        w = int(w_norm * orig_w)
        h = int(h_norm * orig_h)
        
        # Clamp bbox coordinates so they stay on the image
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))
        
        # Classification
        has_frame = bool(class_pred[0][0] > 0.5)
        frame_confidence = float(class_pred[0][0])
        
        # Decode quality flags
        flags = {}
        for i, flag_name in enumerate(FLAG_NAMES):
            flags[flag_name] = bool(flags_pred[0][i] > 0.5)
        
        # Zoom calculation
        frame_area = w * h
        image_area = orig_w * orig_h
        frame_ratio = frame_area / image_area if image_area > 0 else 0
        
        # Compute a simple recommended zoom factor
        ideal_ratio = 0.45
        if frame_ratio > 0:
            recommended_zoom = ideal_ratio / frame_ratio
            recommended_zoom = max(0.5, min(3.0, recommended_zoom))
        else:
            recommended_zoom = 1.0
        
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
                'recommended_zoom': float(recommended_zoom)
            },
            'original_size': {'width': orig_w, 'height': orig_h}
        }
    
    def predict_batch(self, image_paths):
        """Run predict() on a list of image paths and return results.

        This keeps the same API as predict() but operates over a batch for
        speed when possible.
        """
        
        # Preprocess all images
        images = []
        originals = []
        original_sizes = []
        
        for p in image_paths:
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                img = Image.new('RGB', (self.img_width, self.img_height))
            
            orig_w, orig_h = img.size
            img_resized = img.resize((self.img_width, self.img_height))
            img_array = np.array(img_resized) / 255.0
            
            images.append(img_array)
            originals.append(img)
            original_sizes.append((orig_w, orig_h))
        
        batch = np.stack(images, axis=0)
        
        # Predict
        bbox_preds, class_preds, flags_preds = self.model.predict(batch, verbose=0)
        
        results = []
        for i, img_path in enumerate(image_paths):
            orig_w, orig_h = original_sizes[i]
            
            # Denormalize bbox
            x_norm, y_norm, w_norm, h_norm = bbox_preds[i]
            x = int(x_norm * orig_w)
            y = int(y_norm * orig_h)
            w = int(w_norm * orig_w)
            h = int(h_norm * orig_h)
            
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
            
            # Zoom
            frame_area = w * h
            image_area = orig_w * orig_h
            frame_ratio = frame_area / image_area if image_area > 0 else 0
            
            ideal_ratio = 0.45
            if frame_ratio > 0:
                recommended_zoom = ideal_ratio / frame_ratio
                recommended_zoom = max(0.5, min(3.0, recommended_zoom))
            else:
                recommended_zoom = 1.0
            
            results.append({
                'has_frame': has_frame,
                'confidence': frame_confidence,
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                },
                'flags': flags,
                'zoom': {
                    'frame_ratio': float(frame_ratio),
                    'recommended_zoom': float(recommended_zoom)
                }
            })
        
        return results
    
    def test_on_dataset(self, split='test', max_images=None):
        """Test model on a dataset split"""
        dataset_path = os.path.join('./processed_dataset', split)
        
        results = {
            'correct': 0,
            'incorrect': 0,
            'details': []
        }
        
        print(f"\n{'='*60}")
        print(f"Testing on {split.upper()} dataset - Object Detection")
        print(f"{'='*60}")
        
        batch_size = CONFIG.get('batch_size', 32)

        for category in ['frame_present', 'frame_not_present']:
            category_path = os.path.join(dataset_path, category)

            if not os.path.exists(category_path):
                print(f"Warning: {category_path} not found")
                continue

            # Gather images from the split/category folder
            images = list(map(str, Path(category_path).glob('*.jpg')))
            if max_images:
                images = images[:max_images]

            print(f"\n{category.replace('_', ' ').title()}: {len(images)} images")

            # Process in batches
            for i in tqdm(range(0, len(images), batch_size), desc=f"Processing {category}"):
                batch_paths = images[i:i+batch_size]
                batch_results = self.predict_batch(batch_paths)

                for img_path, result in zip(batch_paths, batch_results):
                    # Check if prediction matches ground truth
                    ground_truth = category == 'frame_present'
                    correct = result['has_frame'] == ground_truth

                    results['details'].append({
                        'image': str(img_path),
                        'category': category,
                        'predicted': 'frame' if result['has_frame'] else 'no_frame',
                        'correct': correct,
                        'confidence': result['confidence'],
                        'bbox': result['bbox']
                    })

                    if correct:
                        results['correct'] += 1
                    else:
                        results['incorrect'] += 1
        
        # Calculate accuracy
        total = results['correct'] + results['incorrect']
        accuracy = results['correct'] / total * 100 if total > 0 else 0
        
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total images tested: {total}")
        print(f"Correct predictions: {results['correct']}")
        print(f"Incorrect predictions: {results['incorrect']}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Find errors
        errors = [d for d in results['details'] if not d['correct']]
        if errors:
            print(f"\nMisclassified images ({len(errors)}):")
            for error in errors[:5]:
                print(f"  - {os.path.basename(error['image'])}")
                print(f"    Predicted: {error['predicted']}, Confidence: {error['confidence']:.2%}")
        
        return results
    
    def test_single_image(self, image_path):
        """Run prediction on a single image and save/show an annotated result."""
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"Testing single image: {image_path}")
        print(f"{'='*60}")
        
        result = self.predict(image_path)
        
        print(f"\n=== Detection Results ===")
        print(f"Frame Detected: {'YES' if result['has_frame'] else 'NO'}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print(f"\n=== Bounding Box ===")
        print(f"Position: x={result['bbox']['x']}, y={result['bbox']['y']}")
        print(f"Size: width={result['bbox']['width']}, height={result['bbox']['height']}")
        
        print(f"\n=== Quality Flags ===")
        for flag, value in result['flags'].items():
            status = "⚠ YES" if value else "✓ NO"
            print(f"  {flag.replace('is_', '')}: {status}")
        
        print(f"\n=== Zoom Recommendation ===")
        print(f"Frame covers: {result['zoom']['frame_ratio']:.1%} of image")
        print(f"Recommended zoom: {result['zoom']['recommended_zoom']:.1f}x")
        
        if result['zoom']['frame_ratio'] < 0.2:
            print("→ Move closer to the frame")
        elif result['zoom']['frame_ratio'] > 0.7:
            print("→ Move back from the frame")
        else:
            print("→ Frame distance is good")
        
        if result['flags'].get('is_blurred', False):
            print("→ Image is blurry - hold camera steady")
        if result['flags'].get('is_out_of_bounds', False):
            print("→ Frame extends beyond bounds - reposition")
        
        # Save and optionally display a visualization of the prediction
        self.visualize_prediction(image_path, result)
    
    def test_random_images(self, count=10):
        """Pick a random sample of images across splits and run predictions.

        Useful as a quick spot-check rather than a full evaluation.
        """
        import random
        
        print(f"\n{'='*60}")
        print(f"Testing {count} random images")
        print(f"{'='*60}")
        
        all_images = []
        
        for split in ['train', 'val', 'test']:
            for category in ['frame_present', 'frame_not_present']:
                path = os.path.join('./processed_dataset', split, category)
                if os.path.exists(path):
                    images = list(Path(path).glob('*.jpg'))
                    all_images.extend([(str(img), category) for img in images])
        
        if not all_images:
            print("No images found in processed_dataset")
            return

        # Pick the requested number of random samples (seeded for repeatability)
        random.seed(42)
        sampled = random.sample(all_images, min(count, len(all_images)))
        
        sampled_paths = [p for p, _ in sampled]
        sampled_truths = [truth for _, truth in sampled]

        results = []
        batch_size = CONFIG.get('batch_size', 32)
        for i in range(0, len(sampled_paths), batch_size):
            batch_paths = sampled_paths[i:i+batch_size]
            batch_results = self.predict_batch(batch_paths)

            for img_path, ground_truth, result in zip(batch_paths, sampled_truths[i:i+batch_size], batch_results):
                correct = result['has_frame'] == (ground_truth == 'frame_present')
                results.append({
                    'image': img_path,
                    'ground_truth': ground_truth,
                    'prediction': 'frame' if result['has_frame'] else 'no_frame',
                    'confidence': result['confidence'],
                    'correct': correct
                })

                filename = os.path.basename(img_path)
                status = '✓' if correct else '✗'
                bbox = result['bbox']
                print(f"{status} {filename}: {result['has_frame']} ({result['confidence']:.2%}) "
                      f"BBox: {bbox['width']}x{bbox['height']}")
        
        accuracy = sum(1 for r in results if r['correct']) / len(results) * 100
        print(f"\nRandom sample accuracy: {accuracy:.2f}%")
    
    def visualize_prediction(self, image_path, result):
        """Draw the detected bounding box and flags onto the image and save it.

        Also shows the image with matplotlib if SHOW_PLOTS is True.
        """
        # Load image
        img = Image.open(image_path)
        
        # Draw bounding box
        draw = ImageDraw.Draw(img)
        bbox = result['bbox']
        
        # Choose color based on detection
        color = 'green' if result['has_frame'] else 'orange'
        
        # Draw rectangle
        draw.rectangle(
            [bbox['x'], bbox['y'], 
             bbox['x'] + bbox['width'], bbox['y'] + bbox['height']],
            outline=color,
            width=3
        )
        
        # Add label
        label = f"{'FRAME' if result['has_frame'] else 'NO FRAME'} ({result['confidence']:.1%})"
        draw.text((bbox['x'], bbox['y'] - 20), label, fill=color)
        
        # Show/save the annotated image using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        
        # Add info text
        info_text = f"Frame: {result['has_frame']}\n"
        info_text += f"Confidence: {result['confidence']:.1%}\n"
        info_text += f"BBox: {bbox['width']}x{bbox['height']}\n"
        info_text += f"Zoom: {result['zoom']['recommended_zoom']:.1f}x"
        
        # Add flag warnings
        if result['flags'].get('is_blurred', False):
            info_text += "\n⚠ Blurry"
        if result['flags'].get('is_too_small', False):
            info_text += "\n⚠ Too small"
        if result['flags'].get('is_too_large', False):
            info_text += "\n⚠ Too large"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: prediction_result.png")
        
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test frame detection model (Object Detection)')
    parser.add_argument('--image', '-i', type=str, help='Path to a single image')
    parser.add_argument('--random', '-r', type=int, metavar='N', help='Test N random images')
    parser.add_argument('--all', '-a', action='store_true', help='Test on all test images')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--count', type=int, default=None, 
                       help='Max images to test (use with --all)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots (save only)')
    
    args = parser.parse_args()

    # Respect the --no-show flag
    global SHOW_PLOTS
    if getattr(args, 'no_show', False):
        SHOW_PLOTS = False
    
    print("\n" + "="*60)
    print("FRAME DETECTION MODEL TESTER - OBJECT DETECTION")
    print("="*60)
    
    # Initialize tester
    tester = FrameDetectorTester()
    
    # Run tests based on arguments
    if args.image:
        tester.test_single_image(args.image)
    elif args.random:
        tester.test_random_images(args.random)
    elif args.all:
        tester.test_on_dataset(split=args.split, max_images=args.count)
    else:
        # Default: test on random sample
        print("\nNo specific test requested. Running default tests...")
        print("\n1. Testing 5 random images from dataset...")
        tester.test_random_images(5)


if __name__ == "__main__":
    main()
