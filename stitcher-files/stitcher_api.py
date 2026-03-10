"""
stitcher_api.py

Install:
    pip install fastapi uvicorn python-multipart opencv-python-headless numpy

Run:
    python -m uvicorn stitcher_api:app --host 0.0.0.0 --port 8000 --log-level info
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from beehive_stitcher.stitcher import BeehiveStitcher, fast_config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("stitcher_api")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("panoramas")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Beehive Stitcher API")
app.mount("/panoramas", StaticFiles(directory=OUTPUT_DIR), name="panoramas")


@app.post("/stitch")
async def stitch(
    video: UploadFile = File(...),
    reference: Optional[UploadFile] = File(default=None),
):
    """
    Accept a video upload and an optional reference photo, stitch into a
    panorama, and return the URL.

    - video:     required, the close-up hive recording
    - reference: optional, a wide photo of the entire frame taken before
                 recording, used to align/validate the stitched result

    Response JSON: { "panorama_url": "/panoramas/<id>.jpg" }
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video.")

    logger.info(
        "Request received -- video: '%s', reference: %s",
        video.filename,
        "yes" if reference else "no",
    )

    # ── Write video to temp file ───────────────────────────────────────────────
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_video_path = tmp.name
    logger.info("Video saved to temp file: %s", tmp_video_path)

    # ── Write reference photo to temp file (if provided) ──────────────────────
    tmp_ref_path: Optional[str] = None
    if reference is not None:
        ref_suffix = Path(reference.filename or "ref.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ref_suffix) as tmp_ref:
            shutil.copyfileobj(reference.file, tmp_ref)
            tmp_ref_path = tmp_ref.name
        logger.info("Reference image saved to temp file: %s", tmp_ref_path)

    output_id = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{output_id}.jpg"

    try:
        config = fast_config()
        config.output_path = str(output_path)

        if tmp_ref_path:
            config.reference_image_path = tmp_ref_path
            logger.info("Reference alignment enabled.")

        logger.info("Starting stitcher...")
        stitcher = BeehiveStitcher(config)
        result = stitcher.stitch_video(tmp_video_path)

        logger.info("Stitching complete -- saving output...")
        stitcher.save(result, str(output_path))
        logger.info("Output saved: %s (%dx%d px)", output_path, result.shape[1], result.shape[0])

    except Exception as exc:
        logger.exception("Stitching failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_video_path).unlink(missing_ok=True)
        if tmp_ref_path:
            Path(tmp_ref_path).unlink(missing_ok=True)

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Stitcher produced no output.")

    logger.info("Returning panorama_url: /panoramas/%s.jpg", output_id)
    return {"panorama_url": f"/panoramas/{output_id}.jpg"}


@app.get("/")
def root():
    return {"service": "Beehive Stitcher API", "status": "ok", "endpoints": ["/stitch", "/health"]}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    raise HTTPException(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok"}