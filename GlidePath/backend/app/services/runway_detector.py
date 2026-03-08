# Runway detection service
# ---------------------------------------------------------------------------
# TO REPLACE WITH REAL YOLO:
#   from ultralytics import YOLO
#   model = YOLO("models/runway_yolo.pt")
#   results = model(frame)
# ---------------------------------------------------------------------------


def detect_runway(video_path: str) -> dict:
    """Detect runway in a video using YOLO (stub).

    Args:
        video_path: Absolute path to the saved video file.

    Returns dict with shape:
        {
            "detected":                bool,   # was a runway found?
            "confidence":              float,  # overall detection confidence (0-1)
            "frames_analyzed":         int,    # total frames processed
            "runway_detected_frames":  int,    # frames where runway was found
            "detections": [
                {
                    "frame": int,        # frame index
                    "bbox":  [x1,y1,x2,y2],  # bounding box in pixels
                    "conf":  float,          # per-frame confidence
                },
                ...
            ]
        }
    """
    # --- STUB: replace this block with real YOLO inference ---
    return {
        "detected": True,
        "confidence": 0.89,
        "frames_analyzed": 243,
        "runway_detected_frames": 237,
        "detections": [
            {"frame": 0,  "bbox": [100, 200, 500, 600], "conf": 0.91},
            {"frame": 10, "bbox": [105, 205, 505, 605], "conf": 0.88},
            {"frame": 20, "bbox": [110, 210, 510, 610], "conf": 0.87},
        ],
    }
