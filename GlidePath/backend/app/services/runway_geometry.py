# Runway geometry estimation service
# ---------------------------------------------------------------------------
# TO REPLACE WITH REAL OPENCV:
#   - Convert YOLO bboxes to runway masks
#   - Use cv2.Canny + cv2.HoughLinesP to find runway edges
#   - Fit centerline from left/right edge midpoints
#   - Compute signed lateral offset: runway_center_x - frame_center_x
#     positive = runway is right of center (aircraft drifting left)
#     negative = runway is left of center (aircraft drifting right)
# ---------------------------------------------------------------------------


def estimate_runway_geometry(detections: dict) -> dict:
    """Estimate runway centerline and lateral offset from YOLO detections (stub).

    Args:
        detections: Output dict from detect_runway().

    Returns dict with shape:
        {
            "centerline_detected":  bool,
            "frame_center_x":       float,  # horizontal center of the video frame
            "runway_center_x":      float,  # estimated horizontal center of runway
            "signed_offset_px":     float,  # runway_center_x - frame_center_x
                                            # positive = runway right of center
                                            # negative = runway left of center
            "offset_per_frame":     list[float],  # signed offset for each sampled frame
            "runway_width_px":      float,  # estimated runway width in pixels
        }
    """
    # --- STUB: replace this block with real OpenCV geometry ---
    return {
        "centerline_detected": True,
        "frame_center_x": 640.0,
        "runway_center_x": 658.7,
        "signed_offset_px": 18.7,       # positive = runway is right of frame center
        "offset_per_frame": [12.3, 15.1, 18.4, 20.2, 22.5, 19.8, 17.1],
        "runway_width_px": 450.0,
    }
