Beehive Frame Detection — Minimal Inference README

Purpose
-------
This repository contains a minimal, runnable subset of a beehive frame detection project focused on inference (detecting beehive frames in images and videos). Large datasets, training artifacts, and developer cruft have been removed to keep the repo small and focused on running detection with a pretrained model.



Quick setup (macOS / zsh)
-------------------------
1. Create and activate a fresh virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If you don't want to install full TensorFlow on your machine, you can run inference on another machine or a container with TF available.

Basic usage
-----------
Run fast single/batch image inference (prints JSON results):

```bash
# Single or multiple images
python fast_inference.py --image path/to/image.jpg
python fast_inference.py --image a.jpg b.jpg c.jpg

# Verbose output (more fields)
python fast_inference.py --image path/to/image.jpg --verbose

# Interactive REPL (keep the model loaded for many queries)
python fast_inference.py --interactive
```

Video inference (process frames and optionally save annotated video or JSON):

```bash
# Process every frame, save annotated video
python video_inference.py input.mp4 output_annotated.mp4

# Save results to JSON instead of an annotated video
python video_inference.py input.mp4 --json output.json

# Skip frames (process 1 every N frames)
python video_inference.py input.mp4 output.mp4 --sample 2
```

What the outputs look like
-------------------------
- `fast_inference.py` prints a JSON mapping of input file to detection result(s). Each result contains:
  - `has_frame` (boolean)
  - `confidence` (float 0..1)
  - `bbox` with `x,y,width,height` and normalized coords
  - `flags` (dictionary of quality flags like `is_blurred`, `is_too_small`, etc.)
  - `zoom` recommendations in some utility outputs

- `video_inference.py` can produce:
  - An annotated MP4 with bounding boxes and labels.
  - A JSON file with per-frame detections.

Config and model location
-------------------------
- The repository expects the model to be under `models/frame_detector/`. That directory currently contains checkpoint files (`checkpoint.keras`, `checkpoint.h5`, `model.h5`) and a `saved_model/` export.
- You can override the model directory when running `fast_inference.py` with `--model-dir /path/to/model_dir`.
- If you prefer to use the top-level `test_model.keras`, you can move or copy it into `models/frame_detector/` or change `fast_inference.py` to load it directly.

Notes about training
--------------------
- The original preprocessing and training scripts and the large dataset were removed to keep the repo small. If you want to re-train or re-generate `processed_dataset/`, I'll need to restore `preprocess_data.py` and `train_frame_detector.py` (I can re-add them on request or you can pull them from your git history/backups).

Troubleshooting
---------------
- ``ModuleNotFoundError`` or import errors: ensure you activated the virtualenv and installed dependencies.
- TensorFlow install issues on macOS: prefer `pip install tensorflow-macos` (Apple Silicon) or `pip install tensorflow` (x86). The pinned versions in `requirements.txt` are a reference — you may need platform-specific variants.
- If `fast_inference.py` fails with "No model checkpoint found": verify `models/frame_detector` contains a compatible model file (`checkpoint.keras`, `model.h5` or a SavedModel directory). Use `--model-dir` to point to the correct path.




