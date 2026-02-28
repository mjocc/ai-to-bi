# Quick Reference - Frame Detection Setup

**Your Dataset Location:**
```
/Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI/
├── frame_present/      (images where frame IS visible)
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── frame_not_present/  (images where frame is NOT visible)
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

---

## 🚀 Step-by-Step Workflow

### Step 1: Verify Setup
```bash
python verify_setup.py
```
This checks:
- ✓ Dataset directories exist
- ✓ Images are present
- ✓ Python packages installed
- ✓ All scripts present

### Step 2: Preprocess Data
```bash
python preprocess_data.py
```
This creates:
- `processed_dataset/train/` (70% of data)
- `processed_dataset/val/` (20% of data)
- `processed_dataset/test/` (10% of data)
- Statistics and visualizations

### Step 3: Train Model
```bash
python train_frame_detector.py
```
This produces:
- `models/frame_detector/saved_model/` (TensorFlow format)
- `models/frame_detector/model.h5` (Keras format)
- `models/frame_detector/model.tflite` (Mobile format)
- Training history plots

### Step 4: Test Model
```bash
# Test single image
python test_model.py models/frame_detector path/to/test_image.jpg

# Test folder of images
python test_model.py models/frame_detector path/to/test_folder/

# Interactive mode
python test_model.py
```

---

## 📊 Expected Results

**Good Training Results:**
- Training accuracy: > 90%
- Validation accuracy: > 85%
- Training loss: < 0.3
- Validation loss: < 0.4

**If Results Are Poor:**
1. Add more training images (aim for 200+ per class)
2. Balance classes (equal images in each folder)
3. Remove blurry/mislabeled images
4. Increase epochs (change in config.py)
5. Try different learning rate (change in config.py)

---

## 🔧 Configuration Files

All paths are configured in `config.py`:
```python
DATASET_BASE = "/Users/sophiachan/Desktop/Cambridge/1B/Ticks/AI_TO_BI"
FRAME_PRESENT_DIR = "frame_present"
FRAME_NOT_PRESENT_DIR = "frame_not_present"
```

Training parameters in `train_frame_detector.py`:
```python
CONFIG = {
    'batch_size': 32,      # Reduce to 16 if out of memory
    'epochs': 50,          # Increase for better accuracy
    'learning_rate': 0.0001,  # Adjust if not converging
}
```

---

## 📱 React Native Integration

### After Training:

1. **Copy model to React Native project:**
```bash
mkdir -p MyBeekeepingApp/src/models/frame_detector
cp models/frame_detector/model.json MyBeekeepingApp/src/models/frame_detector/
cp models/frame_detector/group1-shard1of1.bin MyBeekeepingApp/src/models/frame_detector/weights.bin
```

2. **Install dependencies:**
```bash
cd MyBeekeepingApp
npm install
```

3. **Run app:**
```bash
# iOS
npm run ios

# Android
npm run android
```

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Dataset not found" errors
- Check path in `config.py` matches your actual folder location
- Ensure folder names are exactly `frame_present` and `frame_not_present`

### Out of memory during training
- In `config.py`, change `'batch_size': 32` to `'batch_size': 16`
- Or reduce image size: `'img_height': 160, 'img_width': 160`

### Low accuracy
- Add more training images
- Balance the dataset (equal images per class)
- Check images are correctly labeled
- Increase training epochs

### Slow training
- Use GPU if available (TensorFlow will auto-detect)
- Reduce image size
- Reduce batch size

---

## 📞 Quick Commands Reference

```bash
# 1. Setup verification
python verify_setup.py

# 2. Preprocess data
python preprocess_data.py

# 3. Train model
python train_frame_detector.py

# 4. Test model
python test_model.py

# 5. View config
python config.py

# 6. Install dependencies
pip install -r requirements.txt

# 7. Install React Native deps
npm install
```

---

## ✅ Checklist

Before Training:
- [ ] Dataset folders exist at correct path
- [ ] Images are in correct folders (frame_present / frame_not_present)
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] verify_setup.py passes all checks

After Training:
- [ ] Model achieves >85% validation accuracy
- [ ] test_model.py gives correct predictions
- [ ] Model files saved in models/frame_detector/

For React Native:
- [ ] Model copied to React Native project
- [ ] npm install completed
- [ ] App runs on device/simulator
- [ ] Camera permissions configured
- [ ] Predictions work on test images

---

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | All paths and settings in one place |
| `verify_setup.py` | Check everything is configured correctly |
| `preprocess_data.py` | Split and organize your dataset |
| `train_frame_detector.py` | Train the ML model |
| `test_model.py` | Test trained model on new images |
| `FrameDetection.js` | React Native component for the app |
| `IntegrationExample.js` | How to integrate with your full app |
| `requirements.txt` | Python packages needed |
| `package.json` | React Native packages needed |

---

## 🎯 Success Criteria

Your model is ready when:
- ✅ Validation accuracy > 85%
- ✅ Test predictions are correct
- ✅ Confidence scores > 70% for clear images
- ✅ Works on different lighting conditions
- ✅ Works on different frame angles

---

For detailed documentation, see README.md
For step-by-step guide, see QUICKSTART.md
