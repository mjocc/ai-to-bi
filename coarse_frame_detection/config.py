"""
Configuration for the frame detection project.

Keeps the dataset paths and training/preprocessing settings in one place.
Tweak these values if you move the data or want to change training defaults.
"""

import os
from pathlib import Path


# Dataset paths


# Base directory for the dataset. Expects subfolders like
# `frame_present` and `frame_not_present` inside this path.
DATASET_BASE = "/Users/sophiachan/Desktop/Cambridge/1B/Ticks/course_frame_detection"

# Subdirectories
FRAME_PRESENT_DIR = os.path.join(DATASET_BASE, "frame_present")
FRAME_NOT_PRESENT_DIR = os.path.join(DATASET_BASE, "frame_not_present")

# Where preprocessed train/val/test folders will be written
PROCESSED_DATA_DIR = "./processed_dataset"

# Directory where trained model checkpoints are saved/loaded
MODEL_SAVE_DIR = "./models/frame_detector"


# Training defaults. These are sensible starting values and can be adjusted
# for experiments. `dataset_path` should point to the processed dataset that
# contains train/val/test splits.
TRAINING_CONFIG = {
    'dataset_path': "./processed_dataset",

    # Model input resolution
    'img_height': 224,
    'img_width': 224,

    # Training hyperparameters
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.00005,

    # How to split the data
    'validation_split': 0.2,
    'test_split': 0.1,

    # Which features to enable at training time
    'use_detection': True,
    'use_augmentation': True,
    'use_transfer_learning': True,

    # Detection thresholds
    'bbox_iou_threshold': 0.5,
    'bbox_confidence_threshold': 0.5,

    # Relative loss weights; tweak if one objective needs more emphasis
    'loss_weights': {
        'bbox': 1.0,
        'class': 2.0,
        'flags': 1.0
    },

    'model_save_path': MODEL_SAVE_DIR,
    'random_seed': 42,
}


# Preprocessing settings


PREPROCESSING_CONFIG = {
    'source_dir': DATASET_BASE,
    'output_dir': PROCESSED_DATA_DIR,
    'val_split': 0.2,
    'test_split': 0.1,
    'categories': ['frame_present', 'frame_not_present'],
    # If True, expect JSON annotations and produce detection-style outputs
    'use_detection': True,
}
# VALIDATION


def validate_paths():
    """Check that dataset directories exist and report counts.

    Returns True if everything looks ok, False otherwise.
    """
    errors = []
    
    # Check source directories
    if not os.path.exists(DATASET_BASE):
        errors.append(f"Dataset base directory not found: {DATASET_BASE}")
    
    if not os.path.exists(FRAME_PRESENT_DIR):
        errors.append(f"Frame present directory not found: {FRAME_PRESENT_DIR}")
    
    if not os.path.exists(FRAME_NOT_PRESENT_DIR):
        errors.append(f"Frame not present directory not found: {FRAME_NOT_PRESENT_DIR}")
    
    # Count images and JSON files
    if os.path.exists(FRAME_PRESENT_DIR):
        present_count = len([f for f in os.listdir(FRAME_PRESENT_DIR) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        json_count = len([f for f in os.listdir(FRAME_PRESENT_DIR) 
                         if f.lower().endswith('.json')])
        print(f"✓ Frame present images: {present_count}")
        print(f"✓ Frame present annotations: {json_count}")
    
    if os.path.exists(FRAME_NOT_PRESENT_DIR):
        not_present_count = len([f for f in os.listdir(FRAME_NOT_PRESENT_DIR) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"✓ Frame not present images: {not_present_count}")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\n✓ All paths validated successfully!")
    return True

# Helper utilities
def get_image_files(directory):
    """Return a sorted list of image file paths in `directory`."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in valid_extensions:
            files.append(os.path.join(directory, file))
    
    return sorted(files)

def print_dataset_summary():
    """Print a short summary about the dataset contents to stdout."""
    print("=" * 70)
    print("DATASET SUMMARY - OBJECT DETECTION MODE")
    print("=" * 70)
    print(f"\nDataset Location: {DATASET_BASE}")
    
    if os.path.exists(FRAME_PRESENT_DIR):
        present_files = get_image_files(FRAME_PRESENT_DIR)
        json_files = [f for f in os.listdir(FRAME_PRESENT_DIR) if f.endswith('.json')]
        print(f"\nFrame Present:")
        print(f"  Directory: {FRAME_PRESENT_DIR}")
        print(f"  Images: {len(present_files)}")
        print(f"  Annotations (JSON): {len(json_files)}")
        if present_files:
            print(f"  Example: {os.path.basename(present_files[0])}")
    
    if os.path.exists(FRAME_NOT_PRESENT_DIR):
        not_present_files = get_image_files(FRAME_NOT_PRESENT_DIR)
        print(f"\nFrame Not Present:")
        print(f"  Directory: {FRAME_NOT_PRESENT_DIR}")
        print(f"  Images: {len(not_present_files)}")
        if not_present_files:
            print(f"  Example: {os.path.basename(not_present_files[0])}")
    
    print("\n" + "=" * 70)
    print("Detection Features:")
    print("  - Bounding box detection (x, y, width, height)")
    print("  - Frame presence classification")
    print("  - Quality flags: too_small, too_large, blurred, out_of_bounds, rotation")
    print("=" * 70)

if __name__ == "__main__":
    print("Beehive Frame Detection - Dataset Configuration (Object Detection)")
    print()
    
    # Validate paths
    if validate_paths():
        print()
        print_dataset_summary()
        print("\n✓ Configuration is ready!")
        print("\nNext steps:")
        print("  1. Run: python preprocess_data.py")
        print("  2. Run: python train_frame_detector.py")
    else:
        print("\n⚠ Please fix the errors above before proceeding.")

