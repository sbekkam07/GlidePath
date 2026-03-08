# Alignment scoring service
# Classifies approach alignment and stability from runway geometry.

import statistics

# Offset thresholds in pixels
ALIGNED_THRESHOLD_PX = 10.0    # within ±10px = centered
CAUTION_THRESHOLD_PX = 30.0    # within ±30px = caution
# beyond ±30px = warning


def compute_alignment_scores(geometry: dict, frame_count: int, confidence: float) -> dict:
    """Classify approach alignment and stability from runway geometry.

    Args:
        geometry:    Output dict from estimate_runway_geometry().
        frame_count: Total frames in the video.
        confidence:  Overall detection confidence from detect_runway().

    Returns dict with shape matching AnalysisResponse fields (excluding wind):
        {
            "alignment":        str,         # "centered" | "drifting_left" | "drifting_right"
            "stability":        str,         # "stable" | "caution" | "warning"
            "confidence":       float,
            "frame_count":      int,
            "average_offset_px": float,      # absolute average offset
            "offsets":          list[float], # per-frame signed offsets
        }
    """
    # signed_offset_px: positive = runway right of center (aircraft left of centerline)
    #                   negative = runway left of center (aircraft right of centerline)
    signed_offset = geometry.get("signed_offset_px", 0.0)
    offsets = geometry.get("offset_per_frame", [])
    avg_offset = abs(signed_offset)

    # --- Alignment classification (uses sign to distinguish left/right) ---
    if avg_offset <= ALIGNED_THRESHOLD_PX:
        alignment = "centered"
    elif signed_offset > 0:
        # Runway is right of center → aircraft drifting left
        alignment = "drifting_left"
    else:
        # Runway is left of center → aircraft drifting right
        alignment = "drifting_right"

    # --- Stability classification (uses variance across frames) ---
    if len(offsets) >= 2:
        spread = statistics.stdev(offsets)
    else:
        spread = avg_offset

    if avg_offset <= ALIGNED_THRESHOLD_PX and spread < 5.0:
        stability = "stable"
    elif avg_offset <= CAUTION_THRESHOLD_PX:
        stability = "caution"
    else:
        stability = "warning"

    return {
        "alignment": alignment,
        "stability": stability,
        "confidence": confidence,
        "frame_count": frame_count,
        "average_offset_px": round(avg_offset, 2),
        "offsets": offsets,
    }
