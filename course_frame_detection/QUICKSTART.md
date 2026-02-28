# 🚀 Quick Start Guide - Beehive Frame Detection

This guide will take you from raw dataset to deployed mobile app in a few steps.

## 📦 Part 1: Environment Setup

### Step 1: Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: React Native Environment

```bash
# Install Node.js dependencies
npm install

# For iOS (macOS only)
cd ios
pod install
cd ..
```

## 🗂️ Part 2: Data Preparation

### Step 1: Organize Your Dataset

Place your labeled images in this structure:

```
AI_TO_BI/
├── frame_present/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── frame_not_present/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

**Tips for good training data:**
- At least 100-200 images per class
- Variety of lighting conditions
- Different angles and distances
- Mix of frame types and colors
- Include edge cases (partial frames, blurry images)

### Step 2: Preprocess Data

```bash
python preprocess_data.py
```

This will create:
- `processed_dataset/train/` - Training data (70%)
- `processed_dataset/val/` - Validation data (20%)
- `processed_dataset/test/` - Test data (10%)
- `processed_dataset/dataset_stats.json` - Statistics
- Visualization plots

**Check the output:**
- Review `sample_images.png` to verify data quality
- Check `class_distribution.png` for balance
- Examine `dataset_stats.json` for dataset info

## 🧠 Part 3: Model Training

### Step 1: Configure Training

Edit `train_frame_detector.py` if needed:

```python
CONFIG = {
    'dataset_path': './processed_dataset/train',
    'img_height': 224,          # Image size
    'img_width': 224,
    'batch_size': 32,           # Adjust based on GPU memory
    'epochs': 50,               # Maximum epochs
    'learning_rate': 0.0001,    # Learning rate
    'use_augmentation': True,   # Data augmentation
    'use_transfer_learning': True,  # Use MobileNetV2
}
```

### Step 2: Train Model

```bash
python train_frame_detector.py
```

**Expected training time:**
- CPU: 1-2 hours (depending on dataset size)
- GPU: 10-30 minutes

**Output files:**
- `models/frame_detector/saved_model/` - TensorFlow SavedModel
- `models/frame_detector/model.h5` - Keras model
- `models/frame_detector/model.tflite` - TensorFlow Lite (for mobile)
- `models/frame_detector/training_history.png` - Training curves
- `models/frame_detector/config.json` - Training configuration

### Step 3: Validate Model

```bash
# Test on single image
python test_model.py models/frame_detector test_images/test1.jpg

# Test on folder
python test_model.py models/frame_detector test_images/

# Interactive mode
python test_model.py
```

**What to check:**
- Accuracy should be > 85% on validation set
- Predictions should make sense on test images
- Confidence scores should be reasonable (> 0.7 for clear images)

## 📱 Part 4: Mobile Deployment

### Step 1: Copy Model to React Native

```bash
# Create models directory in React Native project
mkdir -p src/models/frame_detector

# Copy model files
cp models/frame_detector/model.json src/models/frame_detector/
cp models/frame_detector/group1-shard1of1.bin src/models/frame_detector/weights.bin
```

### Step 2: Update Model Path

In `FrameDetection.js`, update the model loading:

```javascript
const loadModel = async () => {
  try {
    const modelJson = require('./models/frame_detector/model.json');
    const modelWeights = require('./models/frame_detector/weights.bin');
    const loadedModel = await tf.loadLayersModel(
      bundleResourceIO(modelJson, modelWeights)
    );
    setModel(loadedModel);
  } catch (error) {
    console.error('Model loading failed:', error);
  }
};
```

### Step 3: Run on Device

```bash
# For Android
npm run android

# For iOS
npm run ios
```

## ✅ Part 5: Testing & Validation

### Test Checklist

- [ ] Camera captures images correctly
- [ ] Gallery selection works
- [ ] ML model loads without errors
- [ ] Predictions are accurate (test with known images)
- [ ] UI displays results clearly
- [ ] App works offline
- [ ] Permissions are properly requested
- [ ] Error handling works (test with invalid images)

### Performance Checklist

- [ ] Inference time < 500ms
- [ ] App doesn't crash on low-end devices
- [ ] Memory usage is reasonable
- [ ] Camera preview is smooth
- [ ] Results display quickly

## 🔧 Troubleshooting Common Issues

### Issue: "Module not found" error

**Solution:**
```bash
# Clear cache and reinstall
rm -rf node_modules
npm install
cd ios && pod install && cd ..
```

### Issue: TensorFlow not loading

**Solution:**
```javascript
// Add to your App.js
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

// Before any TF operations
await tf.ready();
```

### Issue: Low accuracy on mobile

**Solutions:**
- Ensure preprocessing matches training (same normalization)
- Check image size matches model input (224x224)
- Verify model was converted correctly
- Test with same images that worked during training

### Issue: Slow inference

**Solutions:**
- Use TFLite instead of TF.js (faster)
- Reduce image size if appropriate
- Use MobileNet or smaller architecture
- Enable GPU acceleration if available

### Issue: Camera permissions denied

**Solution:**
```javascript
import { check, request, PERMISSIONS } from 'react-native-permissions';

const requestCameraPermission = async () => {
  const result = await request(PERMISSIONS.IOS.CAMERA);
  return result === 'granted';
};
```

## 📊 Model Improvement Tips

### If accuracy is low:

1. **Get more data**: Aim for 500+ images per class
2. **Balance classes**: Equal numbers of each class
3. **Improve quality**: Remove blurry or mislabeled images
4. **Add augmentation**: More rotation, scaling, color changes
5. **Try different architectures**: ResNet, EfficientNet
6. **Tune hyperparameters**: Learning rate, batch size, epochs

### If model is too large:

1. **Use quantization**: Reduces model size by 4x
2. **Prune weights**: Remove unnecessary connections
3. **Use smaller architecture**: MobileNetV3, NASNet Mobile
4. **Reduce input size**: 160x160 instead of 224x224

### If inference is slow:

1. **Use TFLite**: Much faster than TF.js
2. **Enable GPU**: Use GPU delegate on mobile
3. **Reduce precision**: Use INT8 quantization
4. **Optimize preprocessing**: Do less image processing

## 🎯 Next Steps

1. **Integrate with main app**: Connect to disease detection module
2. **Add data sync**: Send results to research database
3. **Implement offline mode**: Cache results when offline
4. **Add analytics**: Track usage and accuracy
5. **User testing**: Get feedback from beekeepers
6. **Iterate**: Improve based on real-world usage

## 📚 Additional Resources

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [React Native Vision Camera](https://github.com/mrousavy/react-native-vision-camera)
- [TensorFlow.js React Native](https://www.tensorflow.org/js/tutorials/react-native)
- [Mobile ML Best Practices](https://www.tensorflow.org/lite/performance/best_practices)

## 💬 Need Help?

- Check the main README.md for detailed documentation
- Review code comments in FrameDetection.js
- Test with the interactive test script
- Contact your team members for project integration

## ✨ Success Indicators

You know everything is working when:
- ✅ Model achieves >85% accuracy on test set
- ✅ Mobile app loads model successfully
- ✅ Predictions match test script results
- ✅ Inference time is <500ms
- ✅ UI is responsive and smooth
- ✅ Works offline without errors

Good luck with your Cambridge beekeeping project! 🐝
