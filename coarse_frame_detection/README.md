# Beehive Frame Detection — Inference

This is a trimmed-down version of the frame detection project, just the inference side. The dataset, training scripts, and anything else that wasn't needed for running the model have been removed to keep it small.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't want to install TensorFlow locally, you can run it on another machine or in a container — the scripts don't need anything beyond what's in `requirements.txt`.

## Basic usage

Run inference on one or more images:

```bash
python fast_inference.py --image path/to/image.jpg
python fast_inference.py --image a.jpg b.jpg c.jpg
python fast_inference.py --image path/to/image.jpg --verbose
python fast_inference.py --interactive   # keeps the model loaded, type paths one by one
```

Run inference on a video:

```bash
# annotated output video
python video_inference.py input.mp4 output_annotated.mp4

# save detections to JSON instead
python video_inference.py input.mp4 --json output.json

# only process every other frame
python video_inference.py input.mp4 output.mp4 --sample 2
```

## Output format

`fast_inference.py` prints JSON. Each result has:
- `has_frame` — whether a frame was detected
- `confidence` — model confidence (0–1)
- `bbox` — x, y, width, height in pixels plus normalized versions
- `flags` — quality issues like `is_blurred`, `is_too_small`, etc.
- `zoom` — rough zoom recommendation

`video_inference.py` can write an annotated MP4 and/or a JSON file with per-frame results.

## Model location

The model should be in `models/frame_detector/`. It looks for `checkpoint.keras`, `checkpoint.h5`, `model.h5`, or a `saved_model/` directory in that order. You can point to a different directory with `--model-dir`.

## Notes on training

Training scripts and the dataset aren't included here to keep the repo small. If you need to retrain, `preprocess_data.py` and `train_frame_detector.py` can be pulled back from git history.

## Troubleshooting

- **Import errors** — make sure you activated the venv and ran `pip install -r requirements.txt`
- **TensorFlow on macOS** — use `tensorflow-macos` on Apple Silicon or `tensorflow` on x86; the pinned versions in `requirements.txt` are just a reference
- **"No model checkpoint found"** — check that `models/frame_detector/` has a model file, or pass `--model-dir` to point to the right place
