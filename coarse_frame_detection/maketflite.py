#!/usr/bin/env python3
"""
Converts the trained Keras model to TFLite so it can run on mobile.
Supports a few quantization modes; default just uses the standard optimization.
"""

import os
import sys
import argparse
import numpy as np

# make local imports work when running this file directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras

# fall back to hardcoded defaults if config isn't available
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
    """Try to load the model in several formats and return (model, model_dir).
    Tries .keras, then model.h5, then checkpoint.h5, then SavedModel.
    """
    import tensorflow as tf

    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_dir = os.path.dirname(model_path)

    # 1) .keras is the preferred format for Keras 3
    try:
        print(f"  Trying .keras file: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully from .keras!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output names: {model.output_names}")
        return model, model_dir
    except Exception as e1:
        print(f"  .keras load failed: {e1}")

    # 2) Try model.h5
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

    # 3) Try checkpoint.h5
    checkpoint_h5 = os.path.join(model_dir, 'checkpoint.h5')
    if os.path.exists(checkpoint_h5):
        try:
            print(f"  Trying checkpoint.h5 file: {checkpoint_h5}")
            model = keras.models.load_model(checkpoint_h5, compile=False)
            print("✓ Model loaded successfully from checkpoint.h5!")
            return model, model_dir
        except Exception as e3:
            print(f"  checkpoint.h5 load failed: {e3}")

    # 4) Try SavedModel directory as a last resort
    saved_model_path = os.path.join(model_dir, 'saved_model')
    if os.path.exists(saved_model_path):
        try:
            print(f"  Trying SavedModel format from: {saved_model_path}")
            saved_model = tf.saved_model.load(saved_model_path)

            # wrap the SavedModel serving signature to look like a Keras model
            serving_fn = saved_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

            class SavedModelAsKeras:
                def __init__(self, sm_model, serving_fn):
                    self._model = sm_model
                    self._serving_fn = serving_fn
                    self.inputs = serving_fn.inputs
                    self.outputs = serving_fn.outputs
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
                    """Needed for the TFLite converter to trace the model."""
                    return self._serving_fn(x)

                @property
                def input_shape(self):
                    return self.inputs[0].shape.as_list()

                @property
                def output_names(self):
                    return sorted(self._serving_fn.outputs.keys())

                @property
                def output_shape(self):
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
    
    print(f"\nConverting to TFLite...")
    print(f"  Quantisation mode: {quantization}")

    # use SavedModel directly if we have the path - tends to be more reliable
    if saved_model_path and os.path.exists(saved_model_path):
        print(f"  Using SavedModel format for conversion")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == 'float16':
        converter.target_spec.supported_types = [tf.float16]
        print("  Using float16 quantization")
    elif quantization == 'int8':
        # int8 needs a representative dataset which we don't have here,
        # so just fall back to default optimizations
        print("  Note: Full int8 needs a representative dataset, using default instead")
    elif quantization == 'none':
        converter.optimizations = []
        print("  No quantization (full precision)")
    else:
        print("  Using default optimizations")
    
    # need SELECT_TF_OPS for some ops the model uses
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Saved to: {output_path}")

    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    return tflite_model


def verify_tflite_model(tflite_path):
    """Load the converted model and run a quick dummy inference to check it works."""
    print(f"\nVerifying TFLite model...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input details:")
    for detail in input_details:
        print(f"    - name: {detail['name']}, shape: {detail['shape']}, dtype: {detail['dtype']}")

    print(f"  Output details:")
    for detail in output_details:
        print(f"    - name: {detail['name']}, shape: {detail['shape']}, dtype: {detail['dtype']}")

    # run with random input just to check it doesn't crash
    input_shape = input_details[0]['shape']
    dummy_input = np.random.random(input_shape).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    outputs = []
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        outputs.append(output)

    print(f"  Model runs ok. Output shapes: {[o.shape for o in outputs]}")

    return True


def find_model_checkpoint(model_dir):
    """Scan model_dir for a known checkpoint filename and return the path."""
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

    if args.model:
        model_path = args.model
    else:
        model_path = find_model_checkpoint(model_dir)
        print(f"Found checkpoint: {os.path.basename(model_path)}")

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(model_dir, 'frame_detector.tflite')

    print(f"\nModel dir: {model_dir}")
    print(f"Output: {output_path}")

    # load returns (model, path_used) so we know whether to pass SavedModel path to converter
    try:
        model, saved_model_path = load_keras_model(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        return 1

    try:
        convert_to_tflite(model, output_path, args.quantization, saved_model_path)
    except Exception as e:
        print(f"\nError converting model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if not args.no_verify:
        try:
            verify_tflite_model(output_path)
        except Exception as e:
            print(f"\nVerification failed: {e}")
            print("  The file might still work, but worth checking.")

    print("\nDone!")
    print(f"Copy '{output_path}' to the device and load it with tf.lite.Interpreter")

    return 0


if __name__ == "__main__":
    sys.exit(main())

