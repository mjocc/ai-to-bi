"""
Config file for the frame detection project.

All the paths and training settings live here so I don't have to hunt
through multiple files when something changes. Update DATASET_BASE if
you move the data folder.
"""

import os
from pathlib import Path


# paths to the raw data folders
DATASET_BASE = "/Users/sophiachan/Desktop/Cambridge/1B/Ticks/course_frame_detection"

FRAME_PRESENT_DIR = os.path.join(DATASET_BASE, "frame_present")
FRAME_NOT_PRESENT_DIR = os.path.join(DATASET_BASE, "frame_not_present")

# where the processed train/val/test split ends up
PROCESSED_DATA_DIR = "./processed_dataset"

# where checkpoints get saved during training
MODEL_SAVE_DIR = "./models/frame_detector"


# main training config - tweak batch size / lr / epochs here
TRAINING_CONFIG = {
    'dataset_path': "./processed_dataset",

    'img_height': 224,
    'img_width': 224,

    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.00005,  # lowered from default, seems to work better

    'validation_split': 0.2,
    'test_split': 0.1,

    'use_detection': True,
    'use_augmentation': True,
    'use_transfer_learning': True,  # using MobileNetV2 pretrained weights

    'bbox_iou_threshold': 0.5,
    'bbox_confidence_threshold': 0.5,

    # bumped class weight up because the classification loss was getting swamped
    'loss_weights': {
        'bbox': 1.0,
        'class': 2.0,
        'flags': 1.0
    },

    'model_save_path': MODEL_SAVE_DIR,
    'random_seed': 42,
}


# preprocessing config - mirrors training config but used in preprocess_data.py
PREPROCESSING_CONFIG = {
    'source_dir': DATASET_BASE,
    'output_dir': PROCESSED_DATA_DIR,
    'val_split': 0.2,
    'test_split': 0.1,
    'categories': ['frame_present', 'frame_not_present'],
    'use_detection': True,  # set to True to also parse the JSON annotation files
}
# VALIDATION


def validate_paths():
    """Check the dataset directories exist and print image counts.

    Returns True if all good, False if anything is missing.
    """
    errors = []

    if not os.path.exists(DATASET_BASE):
        errors.append(f"Dataset base directory not found: {DATASET_BASE}")

    if not os.path.exists(FRAME_PRESENT_DIR):
        errors.append(f"Frame present directory not found: {FRAME_PRESENT_DIR}")

    if not os.path.exists(FRAME_NOT_PRESENT_DIR):
        errors.append(f"Frame not present directory not found: {FRAME_NOT_PRESENT_DIR}")

    # count images and annotation files so we know what we're working with
    if os.path.exists(FRAME_PRESENT_DIR):
        present_count = len([f for f in os.listdir(FRAME_PRESENT_DIR)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        json_count = len([f for f in os.listdir(FRAME_PRESENT_DIR)
                         if f.lower().endswith('.json')])
        print(f"frame present images: {present_count}")
        print(f"frame present annotations: {json_count}")

    if os.path.exists(FRAME_NOT_PRESENT_DIR):
        not_present_count = len([f for f in os.listdir(FRAME_NOT_PRESENT_DIR)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"frame not present images: {not_present_count}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\nAll paths look good")
    return True

# Helper utilities
def get_image_files(directory):
    """Return sorted list of image paths inside directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []

    for file in os.listdir(directory):
        if Path(file).suffix.lower() in valid_extensions:
            files.append(os.path.join(directory, file))

    return sorted(files)

def print_dataset_summary():
    """Print a rough summary of what images we have."""
    print("=" * 70)
    print("DATASET SUMMARY")
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

if __name__ == "__main__":
    print("Frame Detection - Dataset Config")
    print()

    if validate_paths():
        print()
        print_dataset_summary()
        print("\nConfig looks good. Next: run preprocess_data.py then train_frame_detector.py")
    else:
        print("\nFix the errors above before continuing.")

