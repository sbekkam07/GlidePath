"""Clean overlay renderer for GlidePath.

Draws a YOLO-style detection box on the runway: thin red outline,
semi-transparent red fill, and a compact "runway 0.87" label above.
Matches the standard Ultralytics detection visualization style.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Any, Dict

# ─── Visual tunables ──────────────────────────────────────────
BBOX_THICKNESS = 1                       # outline thickness (px)
BBOX_COLOR = (0, 0, 255)                 # BGR — red
FILL_ALPHA = 0.35                        # semi-transparent fill opacity

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.55
LABEL_FONT_THICKNESS = 1
LABEL_BG_COLOR = (0, 0, 255)             # BGR — red label background
LABEL_TEXT_COLOR = (255, 255, 255)        # white text
LABEL_PADDING_X = 4
LABEL_PADDING_Y = 4


def render_overlay(
    frame: np.ndarray,
    detection: Dict[str, Any],
    geometry: Dict[str, Any],
    score: Dict[str, Any],
    **_kwargs: Any,
) -> np.ndarray:
    """Composite a YOLO-style detection box onto the original frame.

    Draws a thin red bounding box with a semi-transparent red fill
    and a ``runway <confidence>`` label above the box — matching the
    style shown in standard YOLO detection outputs.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame (not modified in-place).
    detection : dict
        YOLO detection result (``bbox``, ``confidence``).
    geometry : dict
        Geometry result (unused, kept for API compatibility).
    score : dict
        Scoring result (unused, kept for API compatibility).

    Returns
    -------
    np.ndarray
        Copy of frame with overlay composited on top.
    """
    bbox = detection.get("bbox")
    if bbox is None:
        # Nothing detected — return untouched frame
        return frame.copy()

    out = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    confidence = float(detection.get("confidence", 0.0))

    # ── Semi-transparent filled rectangle ─────────────────────
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), BBOX_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, FILL_ALPHA, out, 1.0 - FILL_ALPHA, 0, out)

    # ── Thin outline ──────────────────────────────────────────
    cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS, cv2.LINE_AA)

    # ── Label: "runway 0.87" ──────────────────────────────────
    label = f"runway {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, LABEL_FONT, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)

    # Position label just above the top-left corner of the box
    label_x = x1
    label_y_top = y1 - th - 2 * LABEL_PADDING_Y - baseline
    label_y_bottom = y1

    # If label would go off-screen, place it inside the box instead
    if label_y_top < 0:
        label_y_top = y1
        label_y_bottom = y1 + th + 2 * LABEL_PADDING_Y + baseline

    # Label background
    cv2.rectangle(
        out,
        (label_x, label_y_top),
        (label_x + tw + 2 * LABEL_PADDING_X, label_y_bottom),
        LABEL_BG_COLOR,
        cv2.FILLED,
    )

    # Label text
    text_y = label_y_bottom - LABEL_PADDING_Y - baseline
    cv2.putText(
        out,
        label,
        (label_x + LABEL_PADDING_X, text_y),
        LABEL_FONT,
        LABEL_FONT_SCALE,
        LABEL_TEXT_COLOR,
        LABEL_FONT_THICKNESS,
        cv2.LINE_AA,
    )

    return out
