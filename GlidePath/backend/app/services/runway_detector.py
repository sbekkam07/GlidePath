import logging
import os
from pathlib import Path
from typing import Any, Dict

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime
    YOLO = None
import numpy as np
import cv2

# Project root: GlidePath/GlidePath
BASE_DIR = Path(__file__).resolve().parents[3]

# Model weights default: GlidePath/GlidePath/models/best.pt
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best.pt"
_model_path_raw = os.getenv("GLIDEPATH_MODEL_PATH", str(DEFAULT_MODEL_PATH))
MODEL_PATH = Path(_model_path_raw).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BASE_DIR / MODEL_PATH).resolve()

# Test image: GlidePath/GlidePath/backend/experiments/test1.jpeg
TEST_IMAGE_PATH = BASE_DIR / "backend" / "experiments" / "test1.jpeg"

# Output image: GlidePath/GlidePath/backend/experiments/detector_test_output.png
OUTPUT_IMAGE_PATH = BASE_DIR / "backend" / "experiments" / "detector_test_output.png"

LOGGER = logging.getLogger(__name__)
_MODEL = None
_MODEL_LOAD_FAILED = False


def _empty_detection() -> Dict[str, Any]:
    return {"bbox": None, "confidence": 0.0}


def _get_model():
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None

    if YOLO is None:
        LOGGER.error("ultralytics is not installed; runway detection disabled.")
        _MODEL_LOAD_FAILED = True
        return None

    if not MODEL_PATH.exists():
        LOGGER.error("Model weights not found at %s; runway detection disabled.", MODEL_PATH)
        _MODEL_LOAD_FAILED = True
        return None

    try:
        _MODEL = YOLO(str(MODEL_PATH))
    except Exception:
        LOGGER.exception("Failed to load YOLO model from %s", MODEL_PATH)
        _MODEL_LOAD_FAILED = True
        return None

    return _MODEL


def detect_runway(image: np.ndarray) -> Dict[str, Any]:
    """
    Detect the runway in an OpenCV image using a YOLO model.

    Args:
        image: OpenCV image as a numpy array.

    Returns:
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float
        }
        or
        {
            "bbox": None,
            "confidence": 0.0
        }
    """
    if not isinstance(image, np.ndarray) or image.size == 0:
        return _empty_detection()

    model = _get_model()
    if model is None:
        return _empty_detection()

    try:
        results = model(image)
    except Exception:
        LOGGER.exception("YOLO inference failed for frame.")
        return _empty_detection()

    if not results:
        return _empty_detection()

    best_conf = 0.0
    best_box = None

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()

        for box, conf in zip(xyxy, confs):
            conf = float(conf)
            if conf > best_conf:
                best_conf = conf
                best_box = box

    if best_box is None:
        return _empty_detection()

    x1, y1, x2, y2 = [int(round(v)) for v in best_box.tolist()]

    return {
        "bbox": [x1, y1, x2, y2],
        "confidence": best_conf,
    }


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    image = cv2.imread(str(TEST_IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Test image not found or unreadable: {TEST_IMAGE_PATH}")

    detection = detect_runway(image)
    print(detection)

    if detection["bbox"] is not None:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite(str(OUTPUT_IMAGE_PATH), image)
    print(f"Saved visualization to: {OUTPUT_IMAGE_PATH}")
