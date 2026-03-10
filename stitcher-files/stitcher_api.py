"""
stitcher_api.py

Install:
    pip install fastapi uvicorn python-multipart opencv-python-headless numpy

Run:
    python -muvicorn stitcher_api:app --host 0.0.0.0 --port 8000

The app will be reachable from Expo at http://<your-machine-LAN-ip>:8000
"""

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import the existing stitcher
from beehive_stitcher.stitcher import BeehiveStitcher, slow_config, fast_config

# Output directory
OUTPUT_DIR = Path("panoramas")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Beehive Stitcher API")

# Serve finished panoramas as static files so the app can load them by URL
app.mount("/panoramas", StaticFiles(directory=OUTPUT_DIR), name="panoramas")


@app.post("/stitch")
async def stitch(video: UploadFile = File(...)):
    """
    Accept a video upload, stitch it into a panorama, and return the URL.

    Response JSON:
        { "panorama_url": "/panoramas/<id>.jpg" }
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    # Write the upload to a temp file
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    output_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{output_id}.jpg"

    try:
        config = fast_config()
        config.output_path = str(output_path)

        stitcher = BeehiveStitcher(config)
        result = stitcher.stitch_video(tmp_path)
        stitcher.save(result, str(output_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Stitcher produced no output.")

    return {"panorama_url": f"/panoramas/{output_id}.jpg"}


@app.get("/health")
def health():
    return {"status": "ok"}