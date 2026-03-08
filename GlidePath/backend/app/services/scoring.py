from typing import Any, Dict


def score_alignment(geometry: Dict[str, Any]) -> Dict[str, Any]:
    image_center_x = geometry.get("image_center_x")
    runway_center_bottom_x = geometry.get("runway_center_bottom_x")
    offset_px = geometry.get("offset_px")

    if runway_center_bottom_x is None or offset_px is None:
        return {
            "alignment": "unknown",
            "stability": "unstable",
            "offset_px": None,
            "offset_ratio": None,
        }

    # normalize by image width-ish scale using image center
    offset_ratio = abs(offset_px) / max(image_center_x, 1)

    # tune these later if needed
    if abs(offset_px) < 20:
        alignment = "aligned"
    elif offset_px > 0:
        alignment = "drifting_right"
    else:
        alignment = "drifting_left"

    if offset_ratio < 0.03:
        stability = "stable"
    elif offset_ratio < 0.08:
        stability = "caution"
    else:
        stability = "unstable"

    return {
        "alignment": alignment,
        "stability": stability,
        "offset_px": offset_px,
        "offset_ratio": offset_ratio,
    }