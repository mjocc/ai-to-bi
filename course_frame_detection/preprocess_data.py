"""
Data Preprocessing Script for Beehive Frame Detection - Object Detection Mode

This script prepares your dataset for object detection training by:
1. Parsing JSON annotations to extract bounding boxes and flags
2. Organizing folder structure
3. Splitting data into train/val/test sets
4. Performing data augmentation (optional)
5. Generating dataset statistics

Dataset Location: /Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import configuration (optional)
try:
    from config import PREPROCESSING_CONFIG, validate_paths, print_dataset_summary
    print("✓ Using configuration from config.py")
    validate_paths()
    CONFIG = PREPROCESSING_CONFIG
except ImportError:
    print("ℹ config.py not found, using default configuration")
    CONFIG = {
        'source_dir': '/Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI',
        'output_dir': './processed_dataset',
        'val_split': 0.2,
        'test_split': 0.1,
        'categories': ['frame_present', 'frame_not_present'],
        'use_detection': True,
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


class DataPreprocessor:
    """Handles preprocessing of the frame detection dataset for object detection"""
    
    def __init__(self, source_dir=None, output_dir=None, val_split=0.2, test_split=0.1):
        # Use provided values or fallback to config
        self.source_dir = Path(source_dir) if source_dir else Path(CONFIG['source_dir'])
        self.output_dir = Path(output_dir) if output_dir else Path(CONFIG['output_dir'])
        self.val_split = val_split
        self.test_split = test_split
        self.use_detection = CONFIG.get('use_detection', True)
        
        # Categories
        self.categories = CONFIG.get('categories', ['frame_present', 'frame_not_present'])
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'frame_present': 0,
            'frame_not_present': 0,
            'images_with_bbox': 0,
            'images_without_bbox': 0,
            'image_sizes': [],
            'splits': {},
            'flag_statistics': {flag: 0 for flag in FLAG_NAMES}
        }
    
    def parse_json_annotation(self, json_path):
        """Parse JSON annotation file to extract bbox and flags"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract bounding box
            bbox = None
            if 'bbox' in data:
                top_left = data['bbox'].get('top_left', [0, 0])
                bottom_right = data['bbox'].get('bottom_right', [0, 0])
                
                x = int(top_left[0])
                y = int(top_left[1])
                width = int(bottom_right[0] - top_left[0])
                height = int(bottom_right[1] - top_left[1])
                
                if width > 0 and height > 0:
                    bbox = [x, y, width, height]
            
            # Extract flags
            flags = {}
            for flag_name in FLAG_NAMES:
                flags[flag_name] = bool(data.get(flag_name, False))
                if flags[flag_name]:
                    self.stats['flag_statistics'][flag_name] += 1
            
            return {
                'bbox': bbox,
                'flags': flags,
                'has_bbox': bbox is not None
            }
            
        except Exception as e:
            print(f"Error parsing {json_path}: {e}")
            return {'bbox': None, 'flags': {}, 'has_bbox': False}
    
    def validate_image(self, image_path):
        """Check if image is valid and can be opened"""
        try:
            with Image.open(image_path) as img:
                img.verify()
                # Reopen to get size (verify() closes the file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    self.stats['image_sizes'].append((width, height))
                return True
        except Exception as e:
            print(f"Invalid image {image_path}: {e}")
            return False
    
    def get_image_annotations(self, image_path):
        """Get JSON annotation for an image"""
        json_path = image_path.with_suffix('.json')
        if json_path.exists():
            return self.parse_json_annotation(json_path)
        return {'bbox': None, 'flags': {}, 'has_bbox': False}
    
    def organize_dataset(self):
        """Organize and validate the source dataset"""
        
        print("Step 1: Validating and organizing dataset...")
        
        valid_files = {
            'frame_present': [],
            'frame_not_present': []
        }
        
        # Store annotations for each image
        self.annotations = {
            'frame_present': {},
            'frame_not_present': {}
        }
        
        for category in self.categories:
            category_path = self.source_dir / category
            
            if not category_path.exists():
                print(f"Warning: Category folder {category} not found!")
                continue
            
            image_files = list(category_path.glob('*.*'))
            print(f"\nProcessing {category}: {len(image_files)} files found")
            
            for img_path in tqdm(image_files, desc=f"Validating {category}"):
                # Skip non-image files
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                    
                self.stats['total_images'] += 1
                
                # Check if valid image
                if self.validate_image(img_path):
                    valid_files[category].append(img_path)
                    self.stats['valid_images'] += 1
                    self.stats[category] += 1
                    
                    # Get annotation
                    if self.use_detection:
                        annotation = self.get_image_annotations(img_path)
                        self.annotations[category][str(img_path)] = annotation
                        
                        if annotation['has_bbox']:
                            self.stats['images_with_bbox'] += 1
                        else:
                            self.stats['images_without_bbox'] += 1
                else:
                    self.stats['invalid_images'] += 1
        
        return valid_files
    
    def split_dataset(self, valid_files):
        """Split dataset into train/val/test sets"""
        
        print("\nStep 2: Splitting dataset...")
        
        splits = {
            'train': {'frame_present': [], 'frame_not_present': []},
            'val': {'frame_present': [], 'frame_not_present': []},
            'test': {'frame_present': [], 'frame_not_present': []}
        }
        
        for category in self.categories:
            files = valid_files[category]
            
            # First split: separate test set
            train_val_files, test_files = train_test_split(
                files,
                test_size=self.test_split,
                random_state=42
            )
            
            # Second split: separate validation from training
            val_size_adjusted = self.val_split / (1 - self.test_split)
            train_files, val_files = train_test_split(
                train_val_files,
                test_size=val_size_adjusted,
                random_state=42
            )
            
            splits['train'][category] = train_files
            splits['val'][category] = val_files
            splits['test'][category] = test_files
            
            # Update statistics
            self.stats['splits'][category] = {
                'train': len(train_files),
                'val': len(val_files),
                'test': len(test_files)
            }
        
        return splits
    
    def copy_files_to_splits(self, splits):
        """Copy files to organized train/val/test directories"""
        
        print("\nStep 3: Copying files to split directories...")
        
        # Create annotations dictionary for output
        all_annotations = []
        image_id = 0
        
        for split_name, categories in splits.items():
            for category, files in categories.items():
                
                # Create directory
                split_dir = self.output_dir / split_name / category
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy files
                for src_file in tqdm(files, desc=f"Copying {split_name}/{category}"):
                    dst_file = split_dir / src_file.name
                    shutil.copy2(src_file, dst_file)
                    
                    # Save annotation if available
                    if self.use_detection and str(src_file) in self.annotations[category]:
                        annotation = self.annotations[category][str(src_file)]
                        
                        if annotation['has_bbox']:
                            # Get image dimensions for normalization
                            with Image.open(src_file) as img:
                                img_width, img_height = img.size
                            
                            # Normalize bbox to [0, 1]
                            x, y, w, h = annotation['bbox']
                            norm_x = x / img_width
                            norm_y = y / img_height
                            norm_w = w / img_width
                            norm_h = h / img_height
                            
                            all_annotations.append({
                                'image_id': image_id,
                                'image_name': src_file.name,
                                'split': split_name,
                                'category': category,
                                'bbox': annotation['bbox'],
                                'bbox_normalized': [norm_x, norm_y, norm_w, norm_h],
                                'flags': annotation['flags']
                            })
                    
                    image_id += 1
        
        # Save annotations to JSON
        if self.use_detection and all_annotations:
            annotations_file = self.output_dir / 'detection_annotations.json'
            with open(annotations_file, 'w') as f:
                json.dump(all_annotations, f, indent=2)
            print(f"\n✓ Detection annotations saved to: {annotations_file}")
    
    def analyze_dataset(self):
        """Generate dataset statistics and visualizations"""
        
        print("\nStep 4: Analyzing dataset...")
        
        # Calculate statistics
        if self.stats['image_sizes']:
            widths = [size[0] for size in self.stats['image_sizes']]
            heights = [size[1] for size in self.stats['image_sizes']]
            
            self.stats['avg_width'] = np.mean(widths)
            self.stats['avg_height'] = np.mean(heights)
            self.stats['min_width'] = np.min(widths)
            self.stats['max_width'] = np.max(widths)
            self.stats['min_height'] = np.min(heights)
            self.stats['max_height'] = np.max(heights)
        
        # Print statistics
        print("\n" + "=" * 60)
        print("Dataset Statistics - Object Detection")
        print("=" * 60)
        print(f"Total images found: {self.stats['total_images']}")
        print(f"Valid images: {self.stats['valid_images']}")
        print(f"Invalid images: {self.stats['invalid_images']}")
        
        if self.use_detection:
            print(f"\nBounding Box Statistics:")
            print(f"  Images with bbox: {self.stats['images_with_bbox']}")
            print(f"  Images without bbox: {self.stats['images_without_bbox']}")
            
            print(f"\nFlag Statistics:")
            for flag, count in self.stats['flag_statistics'].items():
                print(f"  {flag}: {count}")
        
        print(f"\nClass Distribution:")
        print(f"  Frame Present: {self.stats['frame_present']}")
        print(f"  Frame Not Present: {self.stats['frame_not_present']}")
        
        if self.stats['image_sizes']:
            print(f"\nImage Dimensions:")
            print(f"  Average: {self.stats['avg_width']:.0f} x {self.stats['avg_height']:.0f}")
            print(f"  Min: {self.stats['min_width']} x {self.stats['min_height']}")
            print(f"  Max: {self.stats['max_width']} x {self.stats['max_height']}")
        
        print(f"\nData Splits:")
        for category in self.categories:
            if category in self.stats['splits']:
                splits = self.stats['splits'][category]
                print(f"  {category}:")
                print(f"    Train: {splits['train']}")
                print(f"    Val: {splits['val']}")
                print(f"    Test: {splits['test']}")
        
        # Save statistics to JSON
        stats_file = self.output_dir / 'dataset_stats.json'
        # Convert Path objects to strings for JSON serialization
        stats_to_save = {
            k: v for k, v in self.stats.items() 
            if k != 'image_sizes'  # Skip large list
        }
        # Ensure numpy types are converted to native Python types for JSON
        def _json_converter(obj):
            try:
                # numpy integer
                if isinstance(obj, np.integer):
                    return int(obj)
                # numpy floating
                if isinstance(obj, np.floating):
                    return float(obj)
                # numpy arrays
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
            except Exception:
                pass
            # Fallback: let json raise the normal error
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2, default=_json_converter)
        print(f"\nStatistics saved to {stats_file}")
    
    def visualize_samples(self, splits, num_samples=5):
        """Visualize sample images with bounding boxes"""
        
        print("\nStep 5: Creating sample visualizations...")
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for idx, category in enumerate(self.categories):
            train_files = splits['train'][category][:num_samples]
            
            for col, img_path in enumerate(train_files):
                img = Image.open(img_path)
                axes[idx, col].imshow(img)
                axes[idx, col].axis('off')
                
                # Draw bounding box if available
                if self.use_detection and str(img_path) in self.annotations[category]:
                    annotation = self.annotations[category][str(img_path)]
                    if annotation['has_bbox']:
                        x, y, w, h = annotation['bbox']
                        rect = plt.Rectangle((x, y), w, h, 
                                           fill=False, 
                                           color='red', 
                                           linewidth=2)
                        axes[idx, col].add_patch(rect)
                        axes[idx, col].set_title('Frame Detected', 
                                                fontsize=8, color='green')
                
                if col == 0:
                    axes[idx, col].set_ylabel(
                        category.replace('_', ' ').title(),
                        fontsize=12,
                        fontweight='bold'
                    )
        
        plt.suptitle('Sample Images from Dataset (Red = Detected Frame)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        viz_file = self.output_dir / 'sample_images.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to {viz_file}")
        plt.show()
    
    def create_flag_distribution_plot(self):
        """Create bar plot showing flag distribution"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall distribution
        categories_names = ['Frame Present', 'Frame Not Present']
        counts = [self.stats['frame_present'], self.stats['frame_not_present']]
        colors = ['#4CAF50', '#FF9800']
        
        axes[0].bar(categories_names, counts, color=colors)
        axes[0].set_title('Overall Class Distribution', fontweight='bold')
        axes[0].set_ylabel('Number of Images')
        
        for i, count in enumerate(counts):
            axes[0].text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        # Flag distribution
        flag_names = [f.replace('is_', '').replace('_', ' ').title() 
                     for f in FLAG_NAMES]
        flag_counts = list(self.stats['flag_statistics'].values())
        
        axes[1].bar(flag_names, flag_counts, color='#2196F3')
        axes[1].set_title('Quality Flag Distribution', fontweight='bold')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        dist_file = self.output_dir / 'flag_distribution.png'
        plt.savefig(dist_file, dpi=150, bbox_inches='tight')
        print(f"Flag distribution plot saved to {dist_file}")
        plt.show()
    
    def run(self):
        """Execute the complete preprocessing pipeline"""
        
        print("=" * 60)
        print("Beehive Frame Detection - Data Preprocessing")
        print("Object Detection Mode")
        print("=" * 60)
        
        # Validate and organize
        valid_files = self.organize_dataset()
        
        # Split dataset
        splits = self.split_dataset(valid_files)
        
        # Copy to organized structure
        self.copy_files_to_splits(splits)
        
        # Analyze and visualize
        self.analyze_dataset()
        self.create_flag_distribution_plot()
        self.visualize_samples(splits)
        
        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)
        print(f"Organized dataset saved to: {self.output_dir}")


def main():
    """Main execution"""
    
    print("=" * 60)
    print("Beehive Frame Detection - Data Preprocessing")
    print("Object Detection Mode")
    print("=" * 60)
    
    # Use config if available, otherwise use defaults
    SOURCE_DIR = CONFIG.get('source_dir', '/Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI')
    OUTPUT_DIR = CONFIG.get('output_dir', './processed_dataset')
    VAL_SPLIT = CONFIG.get('val_split', 0.2)
    TEST_SPLIT = CONFIG.get('test_split', 0.1)
    
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Validation split: {VAL_SPLIT * 100}%")
    print(f"Test split: {TEST_SPLIT * 100}%")
    print(f"Detection mode: {CONFIG.get('use_detection', True)}")
    print()
    
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()

