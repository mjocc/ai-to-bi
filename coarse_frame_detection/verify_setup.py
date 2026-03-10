#!/usr/bin/env python3
"""
Setup Verification Script

Run this script to verify your dataset is correctly set up before training.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

def check_directory(path, name):
    """Check if a directory exists and count images"""
    if not os.path.exists(path):
        print(f"{RED}✗{RESET} {name}: NOT FOUND")
        print(f"  Expected: {path}")
        return False, 0
    
    # Count image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_count = 0
    
    for file in os.listdir(path):
        if Path(file).suffix.lower() in valid_extensions:
            image_count += 1
    
    if image_count == 0:
        print(f"{YELLOW}⚠{RESET} {name}: Directory exists but NO IMAGES found")
        print(f"  Location: {path}")
        return True, 0
    
    print(f"{GREEN}✓{RESET} {name}: {image_count} images found")
    print(f"  Location: {path}")
    return True, image_count

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        'tensorflow',
        'numpy',
        'PIL',
        'sklearn',
        'matplotlib',
        'tqdm'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"{GREEN}✓{RESET} {package}")
        except ImportError:
            print(f"{RED}✗{RESET} {package}")
            missing.append(package)
    
    return len(missing) == 0, missing

def main():
    """Main verification"""
    print_header("Beehive Frame Detection - Setup Verification")
    
    # Dataset paths
    DATASET_BASE = "/Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI"
    FRAME_PRESENT = os.path.join(DATASET_BASE, "frame_present")
    FRAME_NOT_PRESENT = os.path.join(DATASET_BASE, "frame_not_present")
    
    all_checks_passed = True
    
    # 1. Check dataset directories
    print(f"{BLUE}1. Checking Dataset Directories{RESET}")
    print("-" * 70)
    
    base_exists = os.path.exists(DATASET_BASE)
    if base_exists:
        print(f"{GREEN}✓{RESET} Base directory found: {DATASET_BASE}")
    else:
        print(f"{RED}✗{RESET} Base directory NOT FOUND: {DATASET_BASE}")
        all_checks_passed = False
    
    print()
    
    present_ok, present_count = check_directory(FRAME_PRESENT, "frame_present")
    not_present_ok, not_present_count = check_directory(FRAME_NOT_PRESENT, "frame_not_present")
    
    total_images = present_count + not_present_count
    
    if not (present_ok and not_present_ok):
        all_checks_passed = False
    
    if total_images < 50:
        print(f"\n{YELLOW}⚠ WARNING:{RESET} Only {total_images} total images found.")
        print(f"  Recommended: At least 100 images (50+ per class) for good results.")
    
    # 2. Check Python packages
    print(f"\n{BLUE}2. Checking Python Dependencies{RESET}")
    print("-" * 70)
    
    packages_ok, missing = check_python_packages()
    
    if not packages_ok:
        print(f"\n{RED}Missing packages:{RESET}")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with: {YELLOW}pip install -r requirements.txt{RESET}")
        all_checks_passed = False
    
    # 3. Check file structure
    print(f"\n{BLUE}3. Checking Project Files{RESET}")
    print("-" * 70)
    
    required_files = {
        'train_frame_detector.py': 'Training script',
        'preprocess_data.py': 'Preprocessing script',
        'test_model.py': 'Testing script',
        'config.py': 'Configuration file',
        'requirements.txt': 'Python dependencies'
    }
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"{GREEN}✓{RESET} {filename} - {description}")
        else:
            print(f"{YELLOW}⚠{RESET} {filename} - {description} (optional)")
    
    # Summary
    print_header("Verification Summary")
    
    if all_checks_passed and total_images >= 50:
        print(f"{GREEN}✓ ALL CHECKS PASSED!{RESET}")
        print(f"\nYou're ready to start training!")
        print(f"\nNext steps:")
        print(f"  1. {YELLOW}python preprocess_data.py{RESET}    - Prepare the dataset")
        print(f"  2. {YELLOW}python train_frame_detector.py{RESET} - Train the model")
        print(f"  3. {YELLOW}python test_model.py{RESET}         - Test the model")
        return 0
    
    elif all_checks_passed:
        print(f"{YELLOW}⚠ SETUP MOSTLY COMPLETE{RESET}")
        print(f"\nWarnings:")
        print(f"  - Low image count ({total_images} images)")
        print(f"  - Recommended: 100+ images for better accuracy")
        print(f"\nYou can proceed, but consider adding more training data.")
        return 0
    
    else:
        print(f"{RED}✗ SETUP INCOMPLETE{RESET}")
        print(f"\nPlease fix the issues above before proceeding.")
        print(f"\nCommon fixes:")
        print(f"  - Check dataset path: {DATASET_BASE}")
        print(f"  - Install dependencies: pip install -r requirements.txt")
        print(f"  - Ensure images are in correct folders:")
        print(f"    • {FRAME_PRESENT}")
        print(f"    • {FRAME_NOT_PRESENT}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
