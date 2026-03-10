#!/usr/bin/env python3
"""
TensorFlow Lite Model Converter

Converts the trained Keras model to TensorFlow Lite format for deployment
on mobile and edge devices.

Usage:
    python maketflite.py                          # Default conversion
    python maketflite.py --quantized              # Convert with quantization
    python maketflite.py --model checkpoint.keras # Specify model file
    python maketflite.py --output custom.tflite   # Custom output name
"""

import os
import sys
import argparse
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras

# Import configuration
try:
    from config import TRAINING_CONFIG
    CONFIG = TRAINING_CONFIG
except ImportError:
    CONFIG = {
        'img_height': 224,
        'img_width': 224,
        'model_save_path': './models/frame_detector',
    }


def load_keras_model(model_path):
    """Load the trained Keras model from checkpoint"""
    import tensorflow as tf
    
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Get model directory from the model path
    model_dir = os.path.dirname(model_path)
    
    # First try: Load from .keras file (preferred for Keras 3)
    try:
        print(f"  Trying .keras file: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully from .keras!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output names: {model.output_names}")
        return model, model_dir
    except Exception as e1:
        print(f"  .keras load failed: {e1}")
    
    # Second try: Load the .h5 version
    h5_path = os.path.join(model_dir, 'model.h5')
    if os.path.exists(h5_path):
        try:
            print(f"  Trying .h5 file: {h5_path}")
            model = keras.models.load_model(h5_path, compile=False)
            print("✓ Model loaded successfully from .h5!")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output names: {model.output_names}")
            return model, model_dir
        except Exception as e2:
            print(f"  .h5 load failed: {e2}")
    
    # Third try: Load from checkpoint.h5
    checkpoint_h5 = os.path.join(model_dir, 'checkpoint.h5')
    if os.path.exists(checkpoint_h5):
        try:
            print(f"  Trying checkpoint.h5 file: {checkpoint_h5}")
            model = keras.models.load_model(checkpoint_h5, compile=False)
            print("✓ Model loaded successfully from checkpoint.h5!")
            return model, model_dir
        except Exception as e3:
            print(f"  checkpoint.h5 load failed: {e3}")
    
    # Fourth try: Load from SavedModel format (most compatible)
    saved_model_path = os.path.join(model_dir, 'saved_model')
    if os.path.exists(saved_model_path):
        try:
            print(f"  Trying SavedModel format from: {saved_model_path}")
            # Load as SavedModel and convert to Keras
            saved_model = tf.saved_model.load(saved_model_path)
            
            # Get the serving function
            serving_fn = saved_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            
            # Create a wrapper that behaves like a Keras model
            class SavedModelAsKeras:
                def __init__(self, sm_model, serving_fn):
                    self._model = sm_model
                    self._serving_fn = serving_fn
                    self.inputs = serving_fn.inputs
                    self.outputs = serving_fn.outputs
                    # Get the concrete function for inference
                    self._concrete_func = serving_fn
                    
                def predict(self, x, verbose=0):
                    result = self._serving_fn(x)
                    outputs = []
                    for key in sorted(result.keys()):
                        outputs.append(result[key].numpy())
                    return tuple(outputs)
                
                def __call__(self, x):
                    return self._serving_fn(x)
                
                def call(self, x, training=False):
                    """Required for TFLite conversion"""
                    return self._serving_fn(x)
                
                @property
                def input_shape(self):
                    return self.inputs[0].shape.as_list()
                
                @property
                def output_names(self):
                    return sorted(self._serving_fn.outputs.keys())
                
                @property
                def output_shape(self):
                    """Return output shapes"""
                    shapes = []
                    for output in self.outputs:
                        shapes.append(output.shape.as_list())
                    return shapes
            
            print("✓ Model loaded successfully from SavedModel!")
            return SavedModelAsKeras(saved_model, serving_fn), saved_model_path
        except Exception as e4:
            print(f"  SavedModel load failed: {e4}")
    
    raise Exception("Could not load model with any method. Please ensure you have the correct Keras/TensorFlow version that was used to train the model.")


def convert_to_tflite(model, output_path, quantization='default', saved_model_path=None):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model: Trained Keras model (or SavedModel wrapper)
        output_path: Path to save the .tflite file
        quantization: Quantization mode:
            - 'none': No quantization (larger file, full precision)
            - 'default': Default optimization (smaller, slight precision loss)
            - 'float16': Float16 quantization (half precision)
            - 'int8': Full integer quantization (smallest, requires representative dataset)
        saved_model_path: Optional path to SavedModel directory for direct conversion
    """
    print(f"\nConverting to TensorFlow Lite...")
    print(f"  Quantization mode: {quantization}")
    
    # If we have the SavedModel path, use it directly for better compatibility
    if saved_model_path and os.path.exists(saved_model_path):
        print(f"  Using SavedModel format for conversion")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    else:
        # Create converter from Keras model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply quantization based on mode
    if quantization == 'float16':
        converter.target_spec.supported_types = [tf.float16]
        print("  Using float16 quantization")
    elif quantization == 'int8':
        # Full integer quantization requires representative data
        # For now, we'll use default optimizations
        print("  Note: Full int8 quantization requires representative dataset")
        print("  Using default optimizations instead")
    elif quantization == 'none':
        converter.optimizations = []
        print("  No quantization (full precision)")
    else:
        # Default mode
        print("  Using default optimizations")
    
    # Enable TensorFlow Lite ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Model converted successfully!")
    print(f"  Saved to: {output_path}")
    
    # Print file size
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")
    
    return tflite_model


def verify_tflite_model(tflite_path):
    """Verify the converted TFLite model works correctly"""
    print(f"\nVerifying TFLite model...")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input details:")
    for detail in input_details:
        print(f"    - name: {detail['name']}, shape: {detail['shape']}, dtype: {detail['dtype']}")
    
    print(f"  Output details:")
    for detail in output_details:
        print(f"    - name: {detail['name']}, shape: {detail['shape']}, dtype: {detail['dtype']}")
    
    # Test with dummy input
    input_shape = input_details[0]['shape']
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    # Get output
    outputs = []
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        outputs.append(output)
    
    print(f"  ✓ Model runs successfully!")
    print(f"    Output shapes: {[o.shape for o in outputs]}")
    
    return True


def find_model_checkpoint(model_dir):
    """Find the best model checkpoint in the model directory"""
    possible_names = [
        'checkpoint.keras',
        'checkpoint.h5',
        'new_model.keras',
        'model.h5',
    ]
    
    for name in possible_names:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"No model checkpoint found in: {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert trained Keras model to TensorFlow Lite format'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to model checkpoint (default: auto-detect)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output .tflite file path (default: models/frame_detector/frame_detector.tflite)'
    )
    parser.add_argument(
        '--quantization', '-q',
        type=str,
        choices=['none', 'default', 'float16', 'int8'],
        default='default',
        help='Quantization mode (default: default)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification after conversion'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Model directory (default: from config)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TensorFlow Lite Model Converter")
    print("=" * 60)
    
    # Determine model directory
    model_dir = args.model_dir or CONFIG.get('model_save_path', './models/frame_detector')
    
    # Find model checkpoint
    if args.model:
        model_path = args.model
    else:
        model_path = find_model_checkpoint(model_dir)
        print(f"Auto-detected model: {os.path.basename(model_path)}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(model_dir, 'frame_detector.tflite')
    
    print(f"\nModel directory: {model_dir}")
    print(f"Output path: {output_path}")
    
    # Load the model - now returns tuple of (model, saved_model_path or model_dir)
    try:
        model, saved_model_path = load_keras_model(model_path)
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return 1
    
    # Convert to TFLite
    try:
        convert_to_tflite(model, output_path, args.quantization, saved_model_path)
    except Exception as e:
        print(f"\n✗ Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify the model
    if not args.no_verify:
        try:
            verify_tflite_model(output_path)
        except Exception as e:
            print(f"\n⚠ Warning: Model verification failed: {e}")
            print("  The model may still work, but there could be issues.")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"\nTo use the TFLite model:")
    print(f"  1. Copy '{output_path}' to your device")
    print(f"  2. Use TensorFlow Lite interpreter to run inference")
    print(f"\nExample usage:")
    print(f"  interpreter = tf.lite.Interpreter(model_path='frame_detector.tflite')")
    print(f"  interpreter.allocate_tensors()")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

