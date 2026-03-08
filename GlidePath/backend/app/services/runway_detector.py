from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO
import numpy as np
import cv2

# Project root: GlidePath/GlidePath
BASE_DIR = Path(__file__).resolve().parents[3]

# Model weights: GlidePath/GlidePath/models/best.pt
MODEL_PATH = BASE_DIR / "models" / "best.pt"

# Test image: GlidePath/GlidePath/backend/experiments/test1.jpeg
TEST_IMAGE_PATH = BASE_DIR / "backend" / "experiments" / "test1.jpeg"

# Output image: GlidePath/GlidePath/backend/experiments/detector_test_output.png
OUTPUT_IMAGE_PATH = BASE_DIR / "backend" / "experiments" / "detector_test_output.png"

# Load YOLO model once
_MODEL = YOLO(str(MODEL_PATH))


def _empty_detection() -> Dict[str, Any]:
    return {"bbox": None, "confidence": 0.0}


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

    results = _MODEL(image)

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