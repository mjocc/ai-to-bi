# Beehive Frame Detection - React Native Module

A mobile application module for detecting beehive frames using on-device machine learning. Part of the Cambridge Beekeeping Assistant project.

## 🐝 Features

- **Camera Integration**: Capture photos of beehive frames directly in-app
- **Gallery Support**: Select existing images for analysis
- **ML-Based Detection**: Object detection with bounding boxes + quality flags
- **On-Device Processing**: Fast inference using TensorFlow Lite
- **Quality Assessment**: Detects blurry, too small, too large, out of bounds, rotation issues
- **Zoom Guidance**: Recommends zoom level based on frame size
- **Data Collection**: Automatically save detection results for research
- **User-Friendly UI**: Visual feedback and guidance for beekeepers
- **Offline Support**: Works without internet connection

## 📋 Prerequisites

- Node.js >= 18
- React Native development environment set up
  - For iOS: Xcode 14+, CocoaPods
  - For Android: Android Studio, JDK 11+
- Python 3.8+ (for model training)

## 🚀 Installation

### 1. Install JavaScript Dependencies

```bash
npm install
# or
yarn install
```

### 2. iOS Setup (macOS only)

```bash
cd ios
pod install
cd ..
```

### 3. Android Setup

No additional setup required for Android.

### 4. Camera Permissions

#### iOS (ios/Info.plist)

Add the following permissions:

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to photograph beehive frames</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select frame images</string>
```

#### Android (android/app/src/main/AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

## 🎯 Usage

### Basic Implementation

```javascript
import React from 'react';
import { View } from 'react-native';
import FrameDetection from './FrameDetection';

function App() {
  const handleDetectionComplete = (result) => {
    console.log('Detection result:', result);
    // Handle the detection result
    // result = { hasFrame, confidence, probabilities, timestamp, imageUri }
  };

  const saveToDatabase = async (data) => {
    // Save detection data to your backend/database
    // This helps with research data collection
    console.log('Saving to database:', data);
  };

  return (
    <View style={{ flex: 1 }}>
      <FrameDetection
        onDetectionComplete={handleDetectionComplete}
        saveToDatabase={saveToDatabase}
      />
    </View>
  );
}

export default App;
```

### Detection Result Object

```javascript
{
  hasFrame: boolean,           // True if frame detected
  confidence: number,           // Confidence score (0-1)
  bbox: {
    x: number,                 // Bounding box x coordinate
    y: number,                 // Bounding box y coordinate
    width: number,             // Bounding box width
    height: number,            // Bounding box height
    normalized: {              // Normalized coordinates (0-1)
      x: number,
      y: number,
      width: number,
      height: number
    }
  },
  flags: {
    isFrame90Degrees: boolean,    // Frame is rotated 90 degrees
    isRotationInvalid: boolean,    // Rotation is invalid
    isTooSmall: boolean,           // Frame is too small in image
    isTooLarge: boolean,           // Frame extends beyond bounds
    isOutOfBounds: boolean,        // Frame outside image bounds
    isBlurred: boolean             // Image is blurry
  },
  zoom: {
    frameRatio: number,            // How much of image the frame covers
    recommendedZoom: number,        // Recommended zoom level (0.5x - 3x)
    isOptimal: boolean,            // Frame is at good distance
    needsZoomIn: boolean,          // Should move closer
    needsZoomOut: boolean,         // Should move back
    instructions: string[]         // User-friendly guidance messages
  },
  timestamp: string,          // ISO timestamp
  imageUri: string,           // Local path to image
  imageSize: {
    width: number,
    height: number
  },
  method: string             // 'model' or 'fallback'
}
```

## 🧠 Model Training

### 1. Prepare Your Dataset

Your dataset should be organized as:

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

### 2. Preprocess Data

```bash
python preprocess_data.py
```

This will:
- Validate all images
- Split into train/val/test sets (70%/20%/10%)
- Generate statistics and visualizations
- Create organized folder structure

### 3. Train the Model

```bash
python train_frame_detector.py
```

This will:
- Train a MobileNetV2-based model (optimized for mobile)
- Apply data augmentation
- Use early stopping and learning rate scheduling
- Save models in multiple formats (TF, H5, TFLite)
- Generate training visualizations

### 4. Model Configuration

Edit the CONFIG dictionary in `train_frame_detector.py`:

```python
CONFIG = {
    'dataset_path': './processed_dataset/train',
    'img_height': 224,
    'img_width': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.0001,
    'validation_split': 0.2,
    'use_augmentation': True,
    'use_transfer_learning': True,
}
```

### 5. Deploy Model

After training, copy the model to your React Native project:

```bash
# For TensorFlow.js format
cp -r models/frame_detector/saved_model ./src/models/frame_detector/

# For TFLite (alternative)
cp models/frame_detector/model.tflite ./android/app/src/main/assets/
```

## 📁 Project Structure

```
beehive-frame-detector/
├── FrameDetection.js           # Main React Native component
├── train_frame_detector.py     # Model training script
├── preprocess_data.py          # Data preprocessing
├── package.json                # Dependencies
├── models/                     # Trained models
│   └── frame_detector/
│       ├── model.json
│       ├── weights.bin
│       └── model.tflite
├── AI_TO_BI/                   # Original dataset
│   ├── frame_present/
│   └── frame_not_present/
└── processed_dataset/          # Preprocessed data
    ├── train/
    ├── val/
    └── test/
```

## 🎨 Customization

### UI Styling

The component uses React Native StyleSheet. Customize colors and layout in the `styles` object:

```javascript
const styles = StyleSheet.create({
  primaryButton: {
    backgroundColor: '#F5A623',  // Change primary color
    // ... other styles
  },
  // ... more style customizations
});
```

### Model Parameters

Adjust preprocessing in `preprocessImage()`:

```javascript
const preprocessImage = async (imageUri) => {
  // Change target size to match your model
  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  
  // Adjust normalization
  const normalized = resized.div(255.0);  // [0, 1]
  // or
  const normalized = resized.div(127.5).sub(1);  // [-1, 1]
  
  return normalized.expandDims(0);
};
```

## 🔧 Troubleshooting

### Common Issues

**1. Camera permission denied**
- Check that permissions are properly added to Info.plist (iOS) and AndroidManifest.xml (Android)
- Request permissions at runtime using react-native-permissions

**2. TensorFlow.js not loading**
- Ensure @tensorflow/tfjs-react-native is properly installed
- Run `await tf.ready()` before loading models

**3. Model not found**
- Verify model files are in the correct location
- Check bundleResourceIO path in `loadModel()`

**4. Image preprocessing errors**
- Ensure images are valid format (JPEG, PNG)
- Check image file permissions

**5. Build errors**
- Clean build folders: `cd android && ./gradlew clean`
- For iOS: `cd ios && pod install && cd ..`

## 📊 Performance

- **Inference Time**: ~100-300ms on modern devices
- **Model Size**: ~15MB (MobileNetV2 with compression)
- **Accuracy**: Typically 90-95% (depends on your dataset)
- **Memory**: ~50-100MB during inference

## 🤝 Contributing

This is a group project for Cambridge beekeeping support. Contributions for:
- Disease detection (other team members)
- Data collection interface
- Beekeeper guidance features
- Integration with bee research databases

## 📝 License

MIT License - see LICENSE file

## 🐝 About the Project

This module is part of a larger mobile application supporting Cambridge beekeepers by:
1. **Frame Detection** (this module) - Detecting frames in photos
2. **Disease Recognition** - AI-based disease identification
3. **Data Collection** - Contributing to bee research
4. **Beekeeper Guidance** - Tools and educational resources

## 📧 Contact

For questions or support with this module:
- Project Team: [Your Team Contact]
- Cambridge Beekeeping Association: [Contact Info]

## 🙏 Acknowledgments

- Cambridge Beekeeping Community
- TensorFlow team for mobile ML tools
- React Native Vision Camera contributors
