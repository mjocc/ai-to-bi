"""
Beehive Frame Detection - Object Detection Model Training Script

This script trains an object detection model to detect beehive frames,
predict bounding boxes, and classify quality flags.

Features:
- Bounding box detection (x, y, width, height)
- Frame presence classification
- Quality flags: too_small, too_large, blurred, out_of_bounds, rotation issues

Dataset Structure:
- /Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI/
  - frame_present/
    - 1.jpg, 2.jpg, ...
    - 1.json, 2.json, ... (annotations with bbox and flags)
  - frame_not_present/
    - 1.jpg, 2.jpg, ...
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

# Import configuration (optional - if config.py exists)
try:
    from config import TRAINING_CONFIG
    CONFIG = TRAINING_CONFIG
    print("✓ Using configuration from config.py")
except ImportError:
    print("ℹ config.py not found, using default configuration")
    # Configuration
    CONFIG = {
        'dataset_path': '/Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI',
        'img_height': 224,
        'img_width': 224,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.0001,
        'validation_split': 0.2,
        'test_split': 0.1,
        'model_save_path': './models/frame_detector',
        'use_augmentation': True,
        'use_transfer_learning': True,
        'random_seed': 42,
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

# Set random seeds for reproducibility
np.random.seed(CONFIG['random_seed'])
tf.random.set_seed(CONFIG['random_seed'])


class DetectionDataGenerator(keras.utils.Sequence):
    """Custom data generator for object detection"""
    
    def __init__(self, image_paths, annotations, img_height, img_width, 
                 batch_size=16, shuffle=True, augmentation=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in indices]
        return self._load_batch(batch_paths)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_batch(self, batch_paths):
        images = []
        bbox_targets = []
        class_targets = []
        flag_targets = []
        
        for img_path in batch_paths:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            original_width, original_height = img.size
            img = img.resize((self.img_width, self.img_height))
            img_array = np.array(img) / 255.0
            
            # Get annotation
            annotation = self.annotations.get(str(img_path), {})
            bbox = annotation.get('bbox', [0, 0, 0, 0])
            flags = annotation.get('flags', {})
            
            # Normalize bbox to [0, 1]
            x, y, w, h = bbox
            if original_width > 0 and original_height > 0:
                norm_x = x / original_width
                norm_y = y / original_height
                norm_w = w / original_width
                norm_h = h / original_height
            else:
                norm_x, norm_y, norm_w, norm_h = 0, 0, 0, 0
            
            # Apply augmentation if enabled
            if self.augmentation:
                img_array, norm_x, norm_y, norm_w, norm_h = self._augment(
                    img_array, norm_x, norm_y, norm_w, norm_h
                )
            
            # Has frame (bbox present)
            has_frame = 1.0 if annotation.get('has_bbox', False) else 0.0
            
            # Flags as binary array
            flags_array = np.array([
                int(flags.get(flag_name, False)) for flag_name in FLAG_NAMES
            ], dtype=np.float32)
            
            images.append(img_array)
            bbox_targets.append([norm_x, norm_y, norm_w, norm_h])
            class_targets.append([has_frame])
            flag_targets.append(flags_array)
        
        return (
            np.array(images),
            {
                'bbox': np.array(bbox_targets),
                'has_frame': np.array(class_targets),
                'flags': np.array(flag_targets)
            }
        )
    
    def _augment(self, img, x, y, w, h):
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img)
            x = 1.0 - x - w
        
        # Random brightness
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)
        
        return img, x, y, w, h


def build_detection_model(input_shape=(224, 224, 3), use_transfer_learning=True):
    """
    Build object detection model
    
    Returns:
        model that outputs:
        - bbox: [x, y, w, h] normalized to [0, 1]
        - has_frame: binary classification
        - flags: multi-label classification (6 flags)
    """
    
    if use_transfer_learning:
        # Use MobileNetV2 for efficient mobile deployment
        print("Building detection model with MobileNetV2 transfer learning...")
        
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Shared dense layer
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Bounding box regression head (4 outputs: x, y, w, h)
        bbox_dense = layers.Dense(128, activation='relu')(x)
        bbox_dense = layers.Dropout(0.2)(bbox_dense)
        bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(bbox_dense)
        
        # Frame classification head (1 output: has_frame)
        class_dense = layers.Dense(64, activation='relu')(x)
        class_dense = layers.Dropout(0.2)(class_dense)
        class_output = layers.Dense(1, activation='sigmoid', name='has_frame')(class_dense)
        
        # Quality flags head (6 outputs: multi-label)
        flags_dense = layers.Dense(64, activation='relu')(x)
        flags_dense = layers.Dropout(0.2)(flags_dense)
        flags_output = layers.Dense(6, activation='sigmoid', name='flags')(flags_dense)
        
        model = keras.Model(
            inputs=inputs,
            outputs=[bbox_output, class_output, flags_output]
        )
        
    else:
        # Custom CNN from scratch
        print("Building custom CNN detection model...")
        
        inputs = keras.Input(shape=input_shape)
        
        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output heads
        bbox_output = layers.Dense(4, activation='sigmoid', name='bbox')(x)
        class_output = layers.Dense(1, activation='sigmoid', name='has_frame')(x)
        flags_output = layers.Dense(6, activation='sigmoid', name='flags')(x)
        
        model = keras.Model(
            inputs=inputs,
            outputs=[bbox_output, class_output, flags_output]
        )
    
    return model


def detection_loss(y_true, y_pred):
    """Custom loss for detection"""
    # Smooth L1 loss for bounding box
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(
        diff < 1.0,
        0.5 * diff ** 2,
        diff - 0.5
    )
    return tf.reduce_mean(loss)


def combined_loss(y_true_bbox, y_pred_bbox, 
                  y_true_class, y_pred_class,
                  y_true_flags, y_pred_flags):
    """Combined loss for all outputs"""
    # Bounding box loss
    bbox_loss = detection_loss(y_true_bbox, y_pred_bbox)
    
    # Classification loss (binary crossentropy)
    class_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true_class, y_pred_class)
    )
    
    # Flags loss (binary crossentropy for each flag)
    flags_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true_flags, y_pred_flags)
    )
    
    # Weighted combined loss
    weights = CONFIG.get('loss_weights', {'bbox': 1.0, 'class': 1.0, 'flags': 1.0})
    total_loss = (
        weights['bbox'] * bbox_loss +
        weights['class'] * class_loss +
        weights['flags'] * flags_loss
    )
    
    return total_loss, bbox_loss, class_loss, flags_loss


def load_annotations(data_dir):
    """Load annotations from processed dataset"""
    annotations = {}
    
    annotations_file = Path(data_dir) / 'detection_annotations.json'
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        for ann in data:
            img_name = ann['image_name']
            img_path = str(Path(data_dir) / ann['split'] / ann['category'] / img_name)
            annotations[img_path] = {
                'bbox': ann['bbox'],
                'bbox_normalized': ann['bbox_normalized'],
                'flags': ann['flags'],
                'has_bbox': True
            }
    
    return annotations


def load_dataset(data_dir):
    """Load dataset with annotations"""
    image_paths = []
    annotations = {}
    
    for split in ['train', 'val', 'test']:
        for category in ['frame_present', 'frame_not_present']:
            split_dir = Path(data_dir) / split / category
            if not split_dir.exists():
                continue
            
            for img_file in split_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                
                # Load annotation
                json_file = img_file.with_suffix('.json')
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract bbox
                    bbox = None
                    if 'bbox' in data:
                        top_left = data['bbox'].get('top_left', [0, 0])
                        bottom_right = data['bbox'].get('bottom_right', [0, 0])
                        x = top_left[0]
                        y = top_left[1]
                        w = bottom_right[0] - top_left[0]
                        h = bottom_right[1] - top_left[1]
                        if w > 0 and h > 0:
                            bbox = [x, y, w, h]
                    
                    # Extract flags
                    flags = {}
                    for flag_name in FLAG_NAMES:
                        flags[flag_name] = data.get(flag_name, False)
                    
                    annotations[str(img_file)] = {
                        'bbox': bbox if bbox else [0, 0, 0, 0],
                        'flags': flags,
                        'has_bbox': bbox is not None
                    }
                else:
                    annotations[str(img_file)] = {
                        'bbox': [0, 0, 0, 0],
                        'flags': {flag_name: False for flag_name in FLAG_NAMES},
                        'has_bbox': False
                    }
    
    return image_paths, annotations


class FrameDetectionTrainer:
    """Handles the complete training pipeline for frame detection"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        self.train_generator = None
        self.val_generator = None
        
    def prepare_data_generators(self, dataset_path):
        """Create data generators for training and validation"""
        
        print("\nLoading dataset...")
        image_paths, annotations = load_dataset(dataset_path)
        
        if not image_paths:
            print("No images found in dataset!")
            return None, None
        
        print(f"Found {len(image_paths)} images")
        
        # Split into train and validation
        train_paths, val_paths = train_test_split(
            image_paths,
            test_size=self.config['validation_split'],
            random_state=self.config['random_seed']
        )
        
        # Create data generators
        self.train_generator = DetectionDataGenerator(
            train_paths, annotations,
            self.config['img_height'], self.config['img_width'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            augmentation=self.config.get('use_augmentation', True)
        )
        
        self.val_generator = DetectionDataGenerator(
            val_paths, annotations,
            self.config['img_height'], self.config['img_width'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            augmentation=None
        )
        
        print(f"\n=== Dataset Information ===")
        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        
        return self.train_generator, self.val_generator
    
    def build_model(self, input_shape=(224, 224, 3)):
        """Build the detection model"""
        
        self.model = build_detection_model(
            input_shape=input_shape,
            use_transfer_learning=self.config.get('use_transfer_learning', True)
        )
        
        return self.model
    
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss={
                'bbox': 'mse',  # Mean squared error for bbox
                'has_frame': 'binary_crossentropy',
                'flags': 'binary_crossentropy'
            },
            loss_weights={
                'bbox': self.config.get('loss_weights', {}).get('bbox', 1.0),
                'has_frame': self.config.get('loss_weights', {}).get('class', 1.0),
                'flags': self.config.get('loss_weights', {}).get('flags', 1.0)
            },
            metrics={
                'bbox': ['mae'],
                'has_frame': ['accuracy'],
                'flags': ['accuracy']
            }
        )
        
        print("\n=== Model Summary ===")
        self.model.summary()
    
    def get_callbacks(self):
        """Define training callbacks"""
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                filepath=f"{self.config['model_save_path']}/checkpoint.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=f"{self.config['model_save_path']}/logs"
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator):
        """Train the model"""
        
        print("\n=== Starting Training ===")
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.config['epochs'],
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, train_generator, val_generator, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        
        if not self.config.get('use_transfer_learning', True):
            print("Fine-tuning only applicable for transfer learning models")
            return
        
        print("\n=== Fine-tuning Model ===")
        
        # Unfreeze the base model
        base_model = self.model.layers[1]  # Assuming base_model is second layer
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate'] / 10),
            loss={
                'bbox': 'mse',
                'has_frame': 'binary_crossentropy',
                'flags': 'binary_crossentropy'
            },
            loss_weights={
                'bbox': self.config.get('loss_weights', {}).get('bbox', 1.0),
                'has_frame': self.config.get('loss_weights', {}).get('class', 1.0),
                'flags': self.config.get('loss_weights', {}).get('flags', 1.0)
            },
            metrics={
                'bbox': ['mae'],
                'has_frame': ['accuracy'],
                'flags': ['accuracy']
            }
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return fine_tune_history
    
    def evaluate_model(self, val_generator):
        """Evaluate model performance"""
        
        print("\n=== Model Evaluation ===")
        
        results = self.model.evaluate(val_generator, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        print("\nEvaluation Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history"""
        
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Bounding box loss
        axes[0, 1].plot(self.history.history['bbox_loss'], label='Train BBox Loss')
        axes[0, 1].plot(self.history.history['val_bbox_loss'], label='Val BBox Loss')
        axes[0, 1].set_title('Bounding Box Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Classification accuracy
        axes[1, 0].plot(self.history.history['has_frame_accuracy'], label='Train Class Acc')
        axes[1, 0].plot(self.history.history['val_has_frame_accuracy'], label='Val Class Acc')
        axes[1, 0].set_title('Frame Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Flags accuracy
        axes[1, 1].plot(self.history.history['flags_accuracy'], label='Train Flags Acc')
        axes[1, 1].plot(self.history.history['val_flags_accuracy'], label='Val Flags Acc')
        axes[1, 1].set_title('Quality Flags Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['model_save_path']}/training_history.png", dpi=300)
        print(f"\nTraining history plot saved to {self.config['model_save_path']}/training_history.png")
        plt.show()
    
    def save_model(self, format='tf'):
        """Save the trained model"""
        
        os.makedirs(self.config['model_save_path'], exist_ok=True)
        
        if format == 'tf':
            # Save as TensorFlow SavedModel
            self.model.save(f"{self.config['model_save_path']}/saved_model")
            print(f"\nModel saved to {self.config['model_save_path']}/saved_model")
        
        elif format == 'h5':
            # Save as H5
            self.model.save(f"{self.config['model_save_path']}/model.h5")
            print(f"\nModel saved to {self.config['model_save_path']}/model.h5")
        
        elif format == 'tflite':
            # Convert to TensorFlow Lite for mobile deployment
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = f"{self.config['model_save_path']}/model.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"\nTFLite model saved to {tflite_path}")
        
        # Save training configuration
        config_path = f"{self.config['model_save_path']}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to {config_path}")
    
    def predict(self, image_path):
        """Predict on a single image and return detection results"""
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_width, original_height = img.size
        img_resized = img.resize((self.config['img_width'], self.config['img_height']))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        bbox_pred, class_pred, flags_pred = self.model.predict(img_array, verbose=0)
        
        # Denormalize bbox
        x, y, w, h = bbox_pred[0]
        x = int(x * original_width)
        y = int(y * original_height)
        w = int(w * original_width)
        h = int(h * original_height)
        
        # Parse results
        has_frame = class_pred[0][0] > 0.5
        confidence = float(class_pred[0][0])
        
        flags = {}
        for i, flag_name in enumerate(FLAG_NAMES):
            flags[flag_name] = bool(flags_pred[0][i] > 0.5)
        
        # Calculate zoom level based on frame size
        frame_area = w * h
        image_area = original_width * original_height
        frame_ratio = frame_area / image_area if image_area > 0 else 0
        
        # Recommended zoom (inverse of frame ratio - smaller frame = more zoom needed)
        recommended_zoom = 1.0
        if frame_ratio > 0:
            recommended_zoom = min(3.0, max(0.5, 1.0 / frame_ratio))
        
        return {
            'has_frame': bool(has_frame),
            'confidence': confidence,
            'bbox': {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'normalized': [float(bbox_pred[0][0]), float(bbox_pred[0][1]), 
                              float(bbox_pred[0][2]), float(bbox_pred[0][3])]
            },
            'flags': flags,
            'zoom': {
                'current_ratio': frame_ratio,
                'recommended_zoom': recommended_zoom,
                'is_too_small': flags.get('is_too_small', False),
                'is_too_large': flags.get('is_too_large', False)
            },
            'original_size': {'width': original_width, 'height': original_height}
        }


def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("Beehive Frame Detection - Object Detection Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = FrameDetectionTrainer(CONFIG)
    
    # Prepare data
    print("\nStep 1: Preparing data generators...")
    dataset_path = CONFIG.get('dataset_path', './processed_dataset')
    train_gen, val_gen = trainer.prepare_data_generators(dataset_path)
    
    if train_gen is None:
        print("Failed to prepare data. Exiting.")
        return
    
    # Build model
    print("\nStep 2: Building model architecture...")
    trainer.build_model()
    trainer.compile_model()
    
    # Train model
    print("\nStep 3: Training model...")
    trainer.train(train_gen, val_gen)
    
    # Fine-tune (if using transfer learning)
    if CONFIG.get('use_transfer_learning', True):
        print("\nStep 4: Fine-tuning model...")
        trainer.fine_tune_model(train_gen, val_gen, epochs=20)
    
    # Evaluate
    print("\nStep 5: Evaluating model...")
    metrics = trainer.evaluate_model(val_gen)
    
    # Plot results
    print("\nStep 6: Plotting training history...")
    trainer.plot_training_history()
    
    # Save models in multiple formats
    print("\nStep 7: Saving models...")
    trainer.save_model(format='tf')
    trainer.save_model(format='h5')
    trainer.save_model(format='tflite')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    print(f"  Total Loss: {metrics.get('loss', 0):.4f}")
    print(f"  BBox Loss: {metrics.get('bbox_loss', 0):.4f}")
    print(f"  Class Accuracy: {metrics.get('has_frame_accuracy', 0):.4f}")
    print(f"  Flags Accuracy: {metrics.get('flags_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()

