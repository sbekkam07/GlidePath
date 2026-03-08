"""YOLO-based runway detection utilities.

This module uses the official Ultralytics Python API (`from ultralytics import YOLO`) and
exposes small helpers that are easy to integrate into the existing GlidePath pipeline.

Public helpers:
- load_model(model_path)
- detect_runway_frame(model, frame)
- detect_runway_video(model, video_path, ...)
- draw_runway_bbox(frame, bbox)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


DetectionDict = Dict[str, Any]
RUNWAY_BOX_COLOR = (0, 190, 255)


def _draw_runway_label(frame: np.ndarray, bbox: list[int], confidence: float) -> None:
    x1, y1, _, _ = [int(v) for v in bbox]
    label = f"runway {float(confidence):.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad_x = 6
    pad_y = 3
    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2
    h, w = frame.shape[:2]
    label_x = max(0, min(w - box_w, x1))
    label_y_bottom = y1 - 4
    if label_y_bottom - box_h < 0:
        label_y_bottom = min(h - 1, y1 + box_h + 2)
    top_left = (label_x, label_y_bottom - box_h)
    bottom_right = (label_x + box_w, label_y_bottom)
    cv2.rectangle(frame, top_left, bottom_right, RUNWAY_BOX_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (top_left[0] + pad_x, bottom_right[1] - baseline - pad_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def load_model(model_path: str | Path) -> YOLO:
    """Load a YOLO model from the provided path and return the model object."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt file, got: {model_path.suffix}")

    # Guard against Git LFS pointer files or other non-binary text payloads.
    with model_path.open("rb") as handle:
        head = handle.read(64)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError(
            f"{model_path} appears to be a Git LFS pointer (not the actual .pt weights). "
            "Run `git lfs pull` or replace this file with a real YOLO checkpoint."
        )

    return YOLO(str(model_path))


def _round_and_clip_bbox(
    bbox: np.ndarray,
    frame: np.ndarray,
) -> list[int] | None:
    """Convert an xyxy float array to integer pixels and clip to frame bounds."""
    if bbox.shape != (4,):
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    # Keep ordering valid for downstream center/size calculations.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if x1 == x2 or y1 == y2:
        return None

    return [x1, y1, x2, y2]


def detect_runway_frame(model: YOLO, frame: np.ndarray) -> Optional[DetectionDict]:
    """Run inference on one OpenCV frame and return the top runway detection.

    The returned detection has the shape:
    {
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
        "center_x": int,
        "center_y": int,
        "width": int,
        "height": int,
    }
    If no detection is found, returns None.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None

    try:
        results = model(frame, verbose=False)
    except Exception:
        return None

    if not results:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None

    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None or conf is None:
        return None

    # Move tensors to CPU to avoid device mismatch and make extraction robust.
    xyxy_np = xyxy.cpu().numpy()
    conf_np = conf.cpu().numpy()

    if xyxy_np.size == 0 or conf_np.size == 0:
        return None

    # Highest-confidence detection for this frame.
    best_idx = int(conf_np.argmax())
    raw_bbox = _round_and_clip_bbox(xyxy_np[best_idx], frame)
    if raw_bbox is None:
        return None

    x1, y1, x2, y2 = raw_bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1
    confidence = float(conf_np[best_idx])

    return {
        "confidence": confidence,
        "bbox": raw_bbox,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
    }


def detect_runway_video(
    model: YOLO,
    video_path: str | Path,
    annotated_output_path: str | Path | None = None,
) -> List[DetectionDict]:
    """Run runway detection on every frame of a video.

    Returns a list with one detection per frame that has a runway prediction.
    Each item contains: frame, confidence, bbox, center_x, center_y.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if annotated_output_path is not None and width > 0 and height > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_output_path),
            fourcc,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer = None

    results_out: List[DetectionDict] = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        detection = detect_runway_frame(model, frame)
        if detection is not None:
            output = {
                "frame": frame_index,
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "center_x": detection["center_x"],
                "center_y": detection["center_y"],
            }
            results_out.append(output)

            if writer is not None:
                draw_runway_bbox(frame, detection["bbox"])
                _draw_runway_label(frame, detection["bbox"], float(detection["confidence"]))

        if writer is not None:
            writer.write(frame)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    return results_out


def draw_runway_bbox(frame: np.ndarray, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    """Draw a green bounding box in-place and return the frame."""
    if frame is None or bbox is None:
        return frame
    if len(bbox) != 4:
        return frame

    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = RUNWAY_BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    return frame


if __name__ == "__main__":
    # Example usage inside this repository.
    BASE_DIR = Path(__file__).resolve().parents[3]
    model_path = BASE_DIR / "models" / "yolov8detect_IN_AND_EXTENDED_ODD_best.pt"
    image_path = BASE_DIR / "backend" / "experiments" / "test1.jpeg"
    video_path = BASE_DIR / "backend" / "experiments" / "runway_demo.mp4"

    model = load_model(model_path)

    sample_image = cv2.imread(str(image_path))
    if sample_image is not None:
        detection = detect_runway_frame(model, sample_image)
        print("Frame detection:", detection)
        if detection and detection.get("bbox"):
            draw_runway_bbox(sample_image, detection["bbox"])
            cv2.imwrite(str(BASE_DIR / "backend" / "experiments" / "runway_single_output.jpg"), sample_image)

    if video_path.exists():
        output = BASE_DIR / "backend" / "experiments" / "runway_video_output.mp4"
        detections = detect_runway_video(
            model,
            str(video_path),
            annotated_output_path=str(output),
        )
        print("Saved annotated video to:", output)
        print("Detections found:", len(detections))
