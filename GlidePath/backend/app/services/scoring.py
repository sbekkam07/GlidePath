"""Alignment and stability scoring helpers."""

import statistics


# Offset thresholds in pixels.
ALIGNED_THRESHOLD_PX = 12.0
CAUTION_THRESHOLD_PX = 32.0


def compute_alignment_scores(geometry: dict, frame_count: int, confidence: float) -> dict:
    """Classify approach alignment and stability from runway geometry offsets."""
    offsets = geometry.get("offset_per_frame", [])
    offset_series = [float(v) for v in offsets if isinstance(v, (int, float))]
    signed_offset = float(geometry.get("signed_offset_px", 0.0) or 0.0)
    signed_mean = statistics.mean(offset_series) if offset_series else signed_offset
    mean_abs_offset = (
        statistics.mean(abs(v) for v in offset_series)
        if offset_series
        else abs(signed_mean)
    )

    if abs(signed_mean) <= ALIGNED_THRESHOLD_PX:
        alignment = "aligned"
    elif signed_mean > 0:
        alignment = "drifting_left"
    else:
        alignment = "drifting_right"

    if len(offset_series) >= 2:
        spread = statistics.pstdev(offset_series)
    else:
        spread = abs(signed_mean)

    if mean_abs_offset <= ALIGNED_THRESHOLD_PX and spread <= 5.0:
        stability = "stable"
    elif mean_abs_offset <= CAUTION_THRESHOLD_PX and spread <= 12.0:
        stability = "caution"
    else:
        stability = "unstable"

    return {
        "alignment": alignment,
        "stability": stability,
        "confidence": confidence,
        "frame_count": frame_count,
        "average_offset_px": round(float(mean_abs_offset), 2),
        "offsets": offset_series,
    }
