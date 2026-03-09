"""
Configuration for Sophia's Frame Detection Dataset

This file contains all the paths and settings specific to your dataset.
Edit these values if your paths change.
"""

import os
from pathlib import Path

# ============================================================================
# DATASET PATHS
# ============================================================================

# Your main dataset directory - using local frame_present and frame_not_present folders
DATASET_BASE = "/Users/sophiachan/Desktop/Cambridge/1B/Ticks/course_frame_detection"

# Subdirectories
FRAME_PRESENT_DIR = os.path.join(DATASET_BASE, "frame_present")
FRAME_NOT_PRESENT_DIR = os.path.join(DATASET_BASE, "frame_not_present")

# Output directory for processed data
PROCESSED_DATA_DIR = "./processed_dataset"

# Model save directory
MODEL_SAVE_DIR = "./models/frame_detector"


# TRAINING CONFIGURATION - OBJECT DETECTION


TRAINING_CONFIG = {
    # Dataset - use processed_dataset which has train/val/test splits
    'dataset_path': "./processed_dataset",
    
    # Image settings
    'img_height': 224,
    'img_width': 224,
    
    # Training parameters - OPTIMIZED SETTINGS
    'batch_size': 32,           # Larger batch for better gradients
    'epochs': 100,               # More epochs with early stopping
    'learning_rate': 0.00005,   # Lower learning rate for finer tuning
    
    # Data splits
    'validation_split': 0.2,    # 20% for validation
    'test_split': 0.1,          # 10% for test
    
    # Model settings - OBJECT DETECTION
    'use_detection': True,      # Enable object detection mode
    'use_augmentation': True,   # Apply data augmentation
    'use_transfer_learning': True,  # Use pretrained MobileNetV2
    
    # Detection-specific settings
    'bbox_iou_threshold': 0.5,
    'bbox_confidence_threshold': 0.5,
    
    # Loss weights - INCREASED CLASSIFICATION WEIGHT
    'loss_weights': {
        'bbox': 1.0,          # Bounding box regression loss weight
        'class': 2.0,         # INCREASED - Frame classification loss weight
        'flags': 1.0          # Quality flags loss weight
    },
    
    # Paths
    'model_save_path': MODEL_SAVE_DIR,
    
    # Other
    'random_seed': 42,
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    'source_dir': DATASET_BASE,
    'output_dir': PROCESSED_DATA_DIR,
    'val_split': 0.2,
    'test_split': 0.1,
    'categories': ['frame_present', 'frame_not_present'],
    'use_detection': True,  # Parse JSON annotations for detection
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Validate that all required directories exist"""
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_files(directory):
    """Get all image files in a directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in valid_extensions:
            files.append(os.path.join(directory, file))
    
    return sorted(files)

def print_dataset_summary():
    """Print a summary of the dataset"""
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

# ============================================================================
# MAIN
# ============================================================================

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

