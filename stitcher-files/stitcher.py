"""
Use locally:

python -m beehive_stitcher.stitcher --input video.mp4 --mode fast --output preview.tiff

Ensure the input video is in the same layer as this stitcher.
There is currently no support for local use with the reference image, but that can be added if needed.

Options:
    --mode {fast,slow}      Processing quality mode (default: slow)
                              fast: ~10s, lower quality, aggressive downsampling
                              slow: full quality, no downsampling
    --output PATH           Output file path (default: beehive_panorama.tiff)
    --reference PATH        Optional reference photo of the full frame
    --pattern GLOB          Image glob pattern for directory input (default: *.jpg)
    --sample-rate INT       Extract every Nth frame (default: 3)
    --max-frames INT        Maximum frames to process (default: 300)
    --no-edges              Disable frame boundary detection
    --no-rect               Disable rectangular frame enforcement
    --debug                 Enable verbose debug logging
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


logging.basicConfig(
    format="%(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StitcherConfig:
    # Frame extraction
    frame_sample_rate: int = 3
    max_frames: int = 300
    input_scale: float = 1.0

    # SIFT
    sift_n_features: int = 1500
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    sift_sigma: float = 1.6

    # FLANN
    flann_trees: int = 5
    flann_checks: int = 50
    lowe_ratio: float = 0.7

    # RANSAC
    ransac_reproj_threshold: float = 2.5
    min_match_count: int = 15

    # Geometric constraints
    enforce_rectangular_frame: bool = True
    max_scale_change: float = 1.08
    max_rotation_deg: float = 3.0
    max_translation_ratio: float = 0.6
    max_shear: float = 0.05

    # Edge weighting
    detect_frame_edges: bool = True
    edge_weight: float = 3.0

    # Output
    output_path: str = "beehive_panorama.tiff"
    use_lanczos: bool = True
    blend_width_ratio: float = 0.12

    # Optional reference image path
    reference_image_path: Optional[str] = None


def fast_config() -> StitcherConfig:
    """Aggressive settings targeting ~10s total runtime."""
    return StitcherConfig(
        frame_sample_rate=8,
        max_frames=60,
        input_scale=0.4,
        sift_n_features=500,
        flann_trees=3,
        flann_checks=20,
        lowe_ratio=0.75,
        min_match_count=10,
        detect_frame_edges=False,
        use_lanczos=False,
        blend_width_ratio=0.08,
    )


def slow_config() -> StitcherConfig:
    """Full-quality settings."""
    return StitcherConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Reference Image Alignment
# ──────────────────────────────────────────────────────────────────────────────

class ReferenceAligner:
    """
    Uses a full-frame reference photo to validate and optionally re-warp
    the stitched panorama so it matches the known frame geometry.

    The reference is matched against the panorama with SIFT + RANSAC to find
    a homography.  If the homography is plausible the panorama is warped to
    align with the reference's coordinate space (i.e. the final image is in
    the same perspective as the wide reference shot).
    """

    MIN_MATCHES = 20

    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 100}
        )

    def align(self, panorama: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Return the panorama warped into reference space, or the original
        panorama unchanged if alignment fails.
        """
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        pan_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

        kp_ref, desc_ref = self.sift.detectAndCompute(ref_gray, None)
        kp_pan, desc_pan = self.sift.detectAndCompute(pan_gray, None)

        if desc_ref is None or desc_pan is None:
            logger.warning("Reference alignment: no descriptors found, skipping.")
            return panorama

        if len(desc_ref) < self.MIN_MATCHES or len(desc_pan) < self.MIN_MATCHES:
            logger.warning("Reference alignment: too few keypoints, skipping.")
            return panorama

        raw = self.flann.knnMatch(desc_pan, desc_ref, k=2)
        good = [m for m, n in raw if m.distance < 0.7 * n.distance]

        if len(good) < self.MIN_MATCHES:
            logger.warning(
                f"Reference alignment: only {len(good)} matches (need {self.MIN_MATCHES}), skipping."
            )
            return panorama

        src_pts = np.float32([kp_pan[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good])

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        if H is None:
            logger.warning("Reference alignment: homography estimation failed, skipping.")
            return panorama

        inliers = int(mask.sum()) if mask is not None else 0
        logger.info(f"Reference alignment: {inliers} inliers, warping panorama.")

        rh, rw = reference.shape[:2]
        aligned = cv2.warpPerspective(panorama, H, (rw, rh))
        return aligned


# ──────────────────────────────────────────────────────────────────────────────
# Frame Boundary Detection
# ──────────────────────────────────────────────────────────────────────────────

class FrameBoundaryDetector:
    """Detects wooden frame edges to guide feature matching."""

    def get_boundary_mask(self, image: np.ndarray, border_width: int = 60) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)

        gray_f = np.float32(gray)
        corners = cv2.cornerHarris(gray_f, blockSize=5, ksize=3, k=0.04)
        corner_mask = (corners > 0.01 * corners.max()).astype(np.uint8) * 255

        dist_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        dist_corner = cv2.distanceTransform(255 - corner_mask, cv2.DIST_L2, 3)

        weight = (
            np.exp(-dist_edge / border_width)
            + np.exp(-dist_corner / (border_width * 0.5))
        )
        return np.clip(weight, 0, 3).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Feature Matching
# ──────────────────────────────────────────────────────────────────────────────

class SIFTMatcher:
    """SIFT feature matcher with optional edge-region weighting."""

    def __init__(self, config: StitcherConfig):
        self.config = config
        self.boundary = FrameBoundaryDetector()

        self.sift = cv2.SIFT_create(
            nfeatures=config.sift_n_features,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
            sigma=config.sift_sigma,
        )
        self.flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": config.flann_trees},
            {"checks": config.flann_checks},
        )

    def detect(self, image: np.ndarray):
        """Returns (keypoints, descriptors, edge_mask_or_None)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_mask = None
        weighted_mask = None

        if self.config.detect_frame_edges:
            edge_mask = self.boundary.get_boundary_mask(image)
            weighted_mask = (edge_mask * 255).astype(np.uint8)

        kp, desc = self.sift.detectAndCompute(gray, weighted_mask)
        return kp, desc, edge_mask

    def match(self, kp_a, desc_a, kp_b, desc_b, edge_a=None, edge_b=None):
        """Returns ratio-filtered matches, optionally boosted by edge proximity."""
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return []

        raw = self.flann.knnMatch(desc_a, desc_b, k=2)
        matches = [m for m, n in raw if m.distance < self.config.lowe_ratio * n.distance]

        if len(matches) < self.config.min_match_count:
            return matches

        if self.config.detect_frame_edges and edge_a is not None and edge_b is not None:
            matches = self._boost_by_edge_proximity(matches, kp_a, kp_b, edge_a, edge_b)

        return matches

    def _boost_by_edge_proximity(self, matches, kp_a, kp_b, edge_a, edge_b):
        ha, wa = edge_a.shape
        hb, wb = edge_b.shape

        scored = []
        for m in matches:
            xa, ya = (int(v) for v in kp_a[m.queryIdx].pt)
            xb, yb = (int(v) for v in kp_b[m.trainIdx].pt)

            xa, ya = min(xa, wa - 1), min(ya, ha - 1)
            xb, yb = min(xb, wb - 1), min(yb, hb - 1)

            boost = 1.0 + self.config.edge_weight * (edge_a[ya, xa] + edge_b[yb, xb]) / 6.0
            scored.append((m.distance / boost, m))

        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored]


# ──────────────────────────────────────────────────────────────────────────────
# Transform Estimation
# ──────────────────────────────────────────────────────────────────────────────

class TransformEstimator:
    """
    Estimates a constrained similarity (or affine) transform between frames.
    Rejects transforms that would distort the rectangular beehive frame.
    """

    def __init__(self, config: StitcherConfig):
        self.config = config

    def estimate(self, kp_src, kp_dst, matches) -> Optional[np.ndarray]:
        if len(matches) < self.config.min_match_count:
            return None

        src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches])

        if self.config.enforce_rectangular_frame:
            M, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
                maxIters=3000,
                confidence=0.995,
            )
        else:
            M, mask = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
                maxIters=3000,
                confidence=0.995,
            )

        if M is None:
            return None

        H = np.vstack([M, [0, 0, 1]])

        if not self._is_valid(H):
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        return H if inliers >= self.config.min_match_count else None

    def _is_valid(self, H: np.ndarray) -> bool:
        A = H[:2, :2]
        U, S, Vt = np.linalg.svd(A)
        sx, sy = S

        # Reject non-uniform scaling (stretching)
        if max(sx, sy) / (min(sx, sy) + 1e-8) > 1.05:
            return False

        # Reject extreme zoom
        scale = (sx + sy) / 2
        limit = self.config.max_scale_change
        if not (1.0 / limit < scale < limit):
            return False

        # Reject large rotations
        rot = U @ Vt
        angle_deg = abs(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
        if angle_deg > self.config.max_rotation_deg:
            return False

        # Reject shear
        shear = np.linalg.norm(A - U @ np.diag(S) @ Vt) / (np.linalg.norm(A) + 1e-8)
        if shear > self.config.max_shear:
            return False

        # Reject implausible translation
        tx, ty = H[:2, 2]
        if np.hypot(tx, ty) > 1500 * self.config.max_translation_ratio:
            return False

        return True


# ──────────────────────────────────────────────────────────────────────────────
# Panorama Canvas
# ──────────────────────────────────────────────────────────────────────────────

class PanoramaCanvas:
    """Weighted-average blending canvas with cosine feathering."""

    def __init__(self, config: StitcherConfig):
        self.config = config
        self._canvas: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._cumulative_H = np.eye(3, dtype=np.float64)

    def initialise(self, first_frame: np.ndarray):
        h, w = first_frame.shape[:2]
        canvas_h, canvas_w = h * 8, w * 8
        tx, ty = w * 3, h * 3

        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        self._cumulative_H = T
        self._canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        self._weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        self._blit(first_frame, T)

    def add_frame(self, frame: np.ndarray, H_to_prev: np.ndarray):
        self._cumulative_H = self._cumulative_H @ np.linalg.inv(H_to_prev)
        self._blit(frame, self._cumulative_H)

    def get_result(self) -> np.ndarray:
        result = self._canvas.copy()
        valid = self._weights > 0
        result[valid] /= self._weights[valid, np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return self._autocrop(result)

    def _blit(self, frame: np.ndarray, H: np.ndarray):
        ch, cw = self._canvas.shape[:2]
        fh, fw = frame.shape[:2]
        interp = cv2.INTER_LANCZOS4 if self.config.use_lanczos else cv2.INTER_CUBIC

        warped = cv2.warpPerspective(
            frame.astype(np.float32), H, (cw, ch),
            flags=interp,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        feather = self._cosine_mask(fh, fw)
        warped_mask = cv2.warpPerspective(
            feather, H, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        self._canvas += warped * warped_mask[..., np.newaxis]
        self._weights += warped_mask

    def _cosine_mask(self, h: int, w: int) -> np.ndarray:
        bw = int(w * self.config.blend_width_ratio)
        bh = int(h * self.config.blend_width_ratio)

        xs = np.ones(w, dtype=np.float32)
        ys = np.ones(h, dtype=np.float32)

        if 0 < bw < w // 2:
            taper = np.cos(np.linspace(np.pi / 2, 0, bw)) ** 2
            xs[:bw] = taper
            xs[-bw:] = taper[::-1]

        if 0 < bh < h // 2:
            taper = np.cos(np.linspace(np.pi / 2, 0, bh)) ** 2
            ys[:bh] = taper
            ys[-bh:] = taper[::-1]

        return np.outer(ys, xs).astype(np.float32)

    @staticmethod
    def _autocrop(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        m = 20
        x, y = max(0, x - m), max(0, y - m)
        w = min(image.shape[1] - x, w + 2 * m)
        h = min(image.shape[0] - y, h + 2 * m)
        return image[y:y+h, x:x+w]


# ──────────────────────────────────────────────────────────────────────────────
# Frame Extraction
# ──────────────────────────────────────────────────────────────────────────────

class FrameExtractor:
    def __init__(self, config: StitcherConfig):
        self.config = config

    def from_video(self, path: str) -> list:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video: {total} frames @ {fps:.1f} fps")

        frames, idx = [], 0
        while len(frames) < self.config.max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(self._scale(frame))
            idx += self.config.frame_sample_rate

        cap.release()
        logger.info(f"Loaded {len(frames)} frames from video")
        return frames

    def from_directory(self, directory: str, pattern: str = "*.jpg") -> list:
        paths = sorted(Path(directory).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No {pattern} files in {directory}")

        frames = []
        for i, p in enumerate(paths):
            if i % self.config.frame_sample_rate != 0:
                continue
            if len(frames) >= self.config.max_frames:
                break
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Could not read {p}, skipping")
                continue
            frames.append(self._scale(img))

        logger.info(f"Loaded {len(frames)} frames from {directory}")
        return frames

    def _scale(self, frame: np.ndarray) -> np.ndarray:
        if self.config.input_scale == 1.0:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(
            frame,
            (int(w * self.config.input_scale), int(h * self.config.input_scale)),
            interpolation=cv2.INTER_AREA,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main Stitcher
# ──────────────────────────────────────────────────────────────────────────────

class BeehiveStitcher:
    def __init__(self, config: Optional[StitcherConfig] = None):
        self.config = config or StitcherConfig()
        self.extractor = FrameExtractor(self.config)
        self.matcher = SIFTMatcher(self.config)
        self.estimator = TransformEstimator(self.config)

    def stitch_video(self, path: str) -> np.ndarray:
        return self._run(self.extractor.from_video(path))

    def stitch_directory(self, path: str, pattern: str = "*.jpg") -> np.ndarray:
        return self._run(self.extractor.from_directory(path, pattern))

    def save(self, image: np.ndarray, path: Optional[str] = None) -> str:
        out = path or self.config.output_path
        cv2.imwrite(out, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        logger.info(f"Saved {out} ({image.shape[1]}×{image.shape[0]} px)")
        return out

    def _run(self, frames: list) -> np.ndarray:
        if not frames:
            raise ValueError("No frames to stitch.")

        logger.info(f"Stitching {len(frames)} frames...")

        features = [self.matcher.detect(f) for f in frames]

        canvas = PanoramaCanvas(self.config)
        canvas.initialise(frames[0])

        skipped = 0
        for i in range(1, len(frames)):
            kp_prev, desc_prev, edge_prev = features[i - 1]
            kp_curr, desc_curr, edge_curr = features[i]

            matches = self.matcher.match(
                kp_prev, desc_prev, kp_curr, desc_curr, edge_prev, edge_curr
            )
            H = self.estimator.estimate(kp_prev, kp_curr, matches)

            if H is None:
                logger.warning(f"Frame {i}: no valid transform, skipping")
                skipped += 1
                continue

            canvas.add_frame(frames[i], H)

            if i % 5 == 0:
                logger.info(f"  {i}/{len(frames) - 1} frames stitched")

        logger.info(f"Done. Skipped {skipped}/{len(frames)} frames.")
        panorama = canvas.get_result()

        # ── Reference alignment (optional) ────────────────────────────────────
        if self.config.reference_image_path:
            ref = cv2.imread(self.config.reference_image_path, cv2.IMREAD_COLOR)
            if ref is None:
                logger.warning(
                    f"Could not load reference image: {self.config.reference_image_path}"
                )
            else:
                logger.info("Aligning panorama to reference image...")
                panorama = ReferenceAligner().align(panorama, ref)

        return panorama


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stitch beehive frame video or images into a panorama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, metavar="PATH",
                   help="Video file or directory of images")
    p.add_argument("--mode", choices=["fast", "slow"], default="slow",
                   help="fast≈10s lower quality | slow=full quality (default: slow)")
    p.add_argument("--output", default="beehive_panorama.tiff", metavar="PATH",
                   help="Output file path (default: beehive_panorama.tiff)")
    p.add_argument("--reference", default=None, metavar="PATH",
                   help="Optional reference photo of the entire hive frame")
    p.add_argument("--pattern", default="*.jpg", metavar="GLOB",
                   help="Image glob pattern for directory input (default: *.jpg)")
    p.add_argument("--sample-rate", type=int, metavar="N",
                   help="Extract every Nth frame (overrides mode default)")
    p.add_argument("--max-frames", type=int, metavar="N",
                   help="Max frames to process (overrides mode default)")
    p.add_argument("--no-edges", action="store_true",
                   help="Disable frame boundary detection")
    p.add_argument("--no-rect", action="store_true",
                   help="Disable rectangular frame enforcement")
    p.add_argument("--debug", action="store_true",
                   help="Enable verbose debug logging")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    config = fast_config() if args.mode == "fast" else slow_config()
    config.output_path = args.output

    if args.reference:
        config.reference_image_path = args.reference
    if args.sample_rate:
        config.frame_sample_rate = args.sample_rate
    if args.max_frames:
        config.max_frames = args.max_frames
    if args.no_edges:
        config.detect_frame_edges = False
    if args.no_rect:
        config.enforce_rectangular_frame = False

    logger.info(f"Mode: {args.mode} | Input: {args.input} | Output: {args.output}")

    stitcher = BeehiveStitcher(config)
    input_path = Path(args.input)

    t0 = time.time()
    if input_path.is_dir():
        result = stitcher.stitch_directory(str(input_path), args.pattern)
    elif input_path.is_file():
        result = stitcher.stitch_video(str(input_path))
    else:
        sys.exit(f"Error: {args.input} is not a valid file or directory")

    stitcher.save(result)
    logger.info(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()