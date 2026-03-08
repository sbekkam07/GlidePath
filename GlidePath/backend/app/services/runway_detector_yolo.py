"""YOLO-based runway detection module for GlidePath.

Uses the official Ultralytics Python API to run YOLO inference on
single frames or full videos.  The detection output format is designed
to plug directly into ``estimate_runway_geometry()`` and the rest of
the GlidePath analysis pipeline.

Inference patterns are adapted from the geoffrey-g-delhomme/ultralytics
reference repository (LARD project).  Key ideas taken from that repo:

* Load model once with ``YOLO(weights_path)`` and reuse across calls.
* Call ``model(frame)`` to get a list of *Results* objects.
* Extract boxes via ``result.boxes.xyxy`` / ``result.boxes.conf``.
* Use ``result.plot()`` for quick annotated-frame visualisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model path  (GlidePath/models/best.pt)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]          # …/GlidePath
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "best.pt"


# ── Public type aliases ───────────────────────────────────────────────────
Detection = Dict[str, Any]
"""Single-frame detection dict.

Keys
----
confidence : float
    Model confidence for the highest-scoring runway box.
bbox : list[int] | None
    ``[x1, y1, x2, y2]`` pixel coordinates, or ``None`` if nothing found.
center_x : int | None
    Horizontal centre of the bounding box.
center_y : int | None
    Vertical centre of the bounding box.
width : int | None
    Box width in pixels.
height : int | None
    Box height in pixels.
"""


# ======================================================================== #
#  1.  Model loading                                                        #
# ======================================================================== #

def load_model(model_path: str | Path | None = None) -> YOLO:
    """Load a YOLO runway-detection model.

    Parameters
    ----------
    model_path : str | Path | None
        Path to the ``.pt`` weights file.  Falls back to the project
        default at ``models/best.pt`` when *None*.

    Returns
    -------
    YOLO
        Ready-to-use Ultralytics model object.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist on disk.
    """
    path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {path}")
    logger.info("Loading YOLO model from %s", path)
    model = YOLO(str(path))
    return model


# ======================================================================== #
#  2.  Single-frame detection                                               #
# ======================================================================== #

def _empty_detection() -> Detection:
    """Return a safe *no-detection* dict."""
    return {
        "confidence": 0.0,
        "bbox": None,
        "center_x": None,
        "center_y": None,
        "width": None,
        "height": None,
    }


def detect_runway_frame(model: YOLO, frame: np.ndarray) -> Optional[Detection]:
    """Run YOLO inference on a single BGR frame.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model (from :func:`load_model`).
    frame : np.ndarray
        OpenCV BGR image (H x W x 3, ``uint8``).

    Returns
    -------
    Detection | None
        The highest-confidence runway detection, or *None* when
        nothing is found.

    Notes
    -----
    * Only the **single best** detection is returned (highest
      confidence) because GlidePath analyses one runway at a time.
    * The returned dict is immediately compatible with
      ``estimate_runway_geometry(image, detection)`` — it contains
      the ``"bbox"`` and ``"confidence"`` keys that function reads.
    """
    # ── Input validation ──────────────────────────────────────────────
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        logger.warning("detect_runway_frame: invalid frame input")
        return None

    # ── Inference ─────────────────────────────────────────────────────
    try:
        results = model(frame, verbose=False)
    except Exception:
        logger.exception("YOLO inference failed")
        return None

    if not results:
        return None

    # ── Extract highest-confidence box ────────────────────────────────
    best_conf: float = 0.0
    best_box: Optional[np.ndarray] = None

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # .xyxy  → [N, 4]  (x1, y1, x2, y2)
        # .conf  → [N]     confidence scores
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()

        for box, conf in zip(xyxy, confs):
            conf_f = float(conf)
            if conf_f > best_conf:
                best_conf = conf_f
                best_box = box

    if best_box is None:
        return None

    # ── Build detection dict ──────────────────────────────────────────
    x1, y1, x2, y2 = [int(round(v)) for v in best_box.tolist()]
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    cx = x1 + w // 2
    cy = y1 + h // 2

    return {
        "confidence": best_conf,
        "bbox": [x1, y1, x2, y2],
        "center_x": cx,
        "center_y": cy,
        "width": w,
        "height": h,
    }


# ======================================================================== #
#  3.  Video detection                                                      #
# ======================================================================== #

def detect_runway_video(
    model: YOLO,
    video_path: str | Path,
    *,
    frame_interval: int = 1,
    annotated_output_path: str | Path | None = None,
) -> List[Detection]:
    """Run YOLO detection across every *n*-th frame of a video.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    video_path : str | Path
        Path to the input video file.
    frame_interval : int
        Process every *n*-th frame (default ``1`` = every frame).
    annotated_output_path : str | Path | None
        If provided, write a video with bounding-box overlays to this
        path.  Uses MJPEG codec (widely compatible).

    Returns
    -------
    list[Detection]
        One entry per *processed* frame, each containing:
        ``{"frame", "confidence", "bbox", "center_x", "center_y",
        "width", "height"}``.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "detect_runway_video: %s — %d frames @ %.1f fps (%dx%d)",
        video_path.name, total_frames, fps, frame_w, frame_h,
    )

    # ── Optional annotated output writer ──────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    if annotated_output_path is not None:
        out_path = Path(annotated_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_fps = max(1.0, fps / max(1, frame_interval))
        writer = cv2.VideoWriter(
            str(out_path), fourcc, out_fps, (frame_w, frame_h),
        )
        logger.info("Writing annotated video to %s", out_path)

    # ── Frame loop ────────────────────────────────────────────────────
    detections: List[Detection] = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % max(1, frame_interval) == 0:
            det = detect_runway_frame(model, frame)

            entry: Detection = {
                "frame": frame_idx,
                "confidence": det["confidence"] if det else 0.0,
                "bbox": det["bbox"] if det else None,
                "center_x": det["center_x"] if det else None,
                "center_y": det["center_y"] if det else None,
                "width": det["width"] if det else None,
                "height": det["height"] if det else None,
            }
            detections.append(entry)

            # Write annotated frame
            if writer is not None:
                annotated = frame.copy()
                if det and det["bbox"] is not None:
                    draw_runway_bbox(annotated, det["bbox"])
                writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        logger.info("Annotated video saved (%d frames written)", len(detections))

    logger.info(
        "detect_runway_video complete: %d/%d frames processed, %d detections",
        len(detections), frame_idx,
        sum(1 for d in detections if d["bbox"] is not None),
    )
    return detections


# ======================================================================== #
#  4.  Visualisation helper                                                 #
# ======================================================================== #

def draw_runway_bbox(
    frame: np.ndarray,
    bbox: List[int],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw a bounding box on the frame (in-place) and return it.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate.
    bbox : list[int]
        ``[x1, y1, x2, y2]`` pixel coordinates.
    color : tuple[int, int, int]
        BGR colour for the rectangle (default: green).
    thickness : int
        Line thickness in pixels.
    label : str | None
        Optional text label drawn above the box.

    Returns
    -------
    np.ndarray
        The same *frame* array (modified in-place).
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        # Background rectangle behind text for readability
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - th - baseline - 6)),
            (x1 + tw + 4, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, max(th + 2, y1 - baseline - 2)),
            font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA,
        )

    return frame


# ======================================================================== #
#  5.  Pipeline-compatibility wrapper                                       #
# ======================================================================== #

def detect_runway(
    image: np.ndarray,
    model: Optional[YOLO] = None,
) -> Detection:
    """Drop-in replacement for the simple ``detect_runway()`` used by the
    analysis routes.

    This wraps :func:`detect_runway_frame` so the return value always
    has at least ``{"bbox": ..., "confidence": ...}``, matching the
    contract expected by ``estimate_runway_geometry`` and ``score_alignment``.

    Parameters
    ----------
    image : np.ndarray
        BGR frame.
    model : YOLO | None
        If *None*, a module-level default model is loaded lazily.

    Returns
    -------
    Detection
        Always contains ``"bbox"`` and ``"confidence"`` keys.
    """
    if model is None:
        model = _get_default_model()

    det = detect_runway_frame(model, image)
    if det is None:
        return _empty_detection()
    return det


# ── Lazy-loaded module-level model ────────────────────────────────────────
_default_model: Optional[YOLO] = None


def _get_default_model() -> YOLO:
    """Return (and cache) the default YOLO model."""
    global _default_model
    if _default_model is None:
        _default_model = load_model()
    return _default_model


# ======================================================================== #
#  Example usage                                                            #
# ======================================================================== #

if __name__ == "__main__":
    """Quick smoke-test: detect runway in a test image and video."""
    import sys

    # ── Load model ────────────────────────────────────────────────────
    model = load_model()
    print(f"Model loaded from {_DEFAULT_MODEL_PATH}")

    # ── Single-frame test ─────────────────────────────────────────────
    test_image_path = _PROJECT_ROOT / "backend" / "experiments" / "test1.jpeg"
    if test_image_path.exists():
        img = cv2.imread(str(test_image_path))
        if img is not None:
            det = detect_runway_frame(model, img)
            print(f"\n--- Single-frame detection ---")
            print(f"  confidence : {det['confidence']:.4f}" if det else "  No detection")
            if det and det["bbox"]:
                print(f"  bbox       : {det['bbox']}")
                print(f"  center     : ({det['center_x']}, {det['center_y']})")
                print(f"  size       : {det['width']} x {det['height']}")
                # Annotate and save
                draw_runway_bbox(img, det["bbox"], label=f"{det['confidence']:.2f}")
                out = _PROJECT_ROOT / "backend" / "experiments" / "yolo_detector_output.png"
                cv2.imwrite(str(out), img)
                print(f"  Saved      : {out}")
    else:
        print(f"Test image not found: {test_image_path}")

    # ── Video test (only if path given on CLI) ────────────────────────
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])
        out_video = _PROJECT_ROOT / "backend" / "experiments" / "yolo_detector_output.avi"
        results = detect_runway_video(
            model, video_file,
            frame_interval=5,
            annotated_output_path=out_video,
        )
        detected = sum(1 for r in results if r["bbox"] is not None)
        print(f"\n--- Video detection ---")
        print(f"  Frames processed : {len(results)}")
        print(f"  Detections       : {detected}")
        print(f"  Annotated video  : {out_video}")

    # ── Pipeline compatibility demo ───────────────────────────────────
    print("\n--- Pipeline compatibility ---")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_runway(dummy)
    print(f"  Empty-frame result: bbox={result['bbox']}, conf={result['confidence']}")
    print("  Keys:", list(result.keys()))
    print("  ✓ Compatible with estimate_runway_geometry(image, detection)")
