"""Runway geometry estimation from YOLO bbox + Canny/Hough edge detection.

Key design decisions for accuracy and stability:
- ROI is tightly derived from YOLO bbox (small padding only, NO full-frame fallback)
- Lines are filtered for perspective consistency with the runway corridor
- When edge evidence is weak, the bbox-derived corridor is used instead of wrong edges
- Temporal smoothing is aggressive: slow alpha, tight max-shift, prefer reusing prior geometry
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Line = Tuple[int, int, int, int]
TrackState = Dict[str, Any]

# ──────────────────────────────────────────────────────────────
# Tuning knobs
# ──────────────────────────────────────────────────────────────
ROI_PAD_X_FRAC = 0.18          # horizontal pad around det bbox (fraction of bbox width)
ROI_PAD_TOP_FRAC = 0.10        # top pad (fraction of bbox height)
ROI_PAD_BOTTOM_FRAC = 0.15     # bottom pad

CANNY_LOW = 40
CANNY_HIGH = 120
HOUGH_THRESHOLD = 28
HOUGH_MIN_LENGTH_FRAC = 0.16   # min line length as fraction of ROI height
HOUGH_MAX_GAP = 30

MIN_LINE_LENGTH = 25
MIN_ANGLE_FROM_VERTICAL = 3.0  # degrees - nearly vertical lines are noise
MAX_ANGLE_FROM_VERTICAL = 38.0 # too-slanted lines are not runway edges
MIN_VERTICAL_SPAN_FRAC = 0.28  # minimum vertical span as fraction of ROI height
LATERAL_TOLERANCE_FRAC = 0.20  # candidate must be within this fraction of ROI width from center

MIN_PAIR_SCORE = 0.22          # pair must score above this to be accepted
VP_X_TOLERANCE = 0.35          # vanishing point must be within this fraction of ROI width from center
VP_Y_MAX = 0.55                # vanishing point must be above this fraction of ROI height

SMOOTH_ALPHA = 0.20            # low alpha = more smoothing (slower response)
MAX_SHIFT_FRAC = 0.035         # maximum per-frame shift as fraction of image width
REUSE_PREV_THRESHOLD = 0.22    # if current pair score is below this, consider reusing previous
REUSE_PREV_SCORE_REQ = 0.22    # previous state must have scored >= this to be reusable
JUMP_THRESHOLD_FRAC = 0.08     # if smoothed line shifts more than this fraction, reject and reuse prior
ANGLE_JUMP_MAX = 10.0          # degrees - max angle change between frames


# ──────────────────────────────────────────────────────────────
# Debug dataclass
# ──────────────────────────────────────────────────────────────
@dataclass
class GeometryDebug:
    roi_source: str = "none"
    roi_box: Optional[List[int]] = None
    raw_line_count: int = 0
    left_candidate_count: int = 0
    right_candidate_count: int = 0
    pair_score: float = -1.0
    pairing_reason: str = "not_tried"
    candidate_reject_count: Dict[str, int] = field(default_factory=dict)
    selected_left: Optional[Line] = None
    selected_right: Optional[Line] = None
    smoothed_from_previous: bool = False
    reused_geometry: bool = False


# ──────────────────────────────────────────────────────────────
# Small geometry utilities
# ──────────────────────────────────────────────────────────────

def _safe_bbox(box, w, h):
    """Clamp a bounding box to image bounds."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def _clamp_line(line):
    return int(round(line[0])), int(round(line[1])), int(round(line[2])), int(round(line[3]))


def _line_length(line):
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


def _line_x_at_y(line, y):
    """Interpolate x position of line at a given y."""
    x1, y1, x2, y2 = line
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return None
    t = (y - y1) / dy
    return x1 + t * (x2 - x1)


def _line_top_bottom(line):
    """Reorder so y1 <= y2 (top is first)."""
    x1, y1, x2, y2 = line
    return (x1, y1, x2, y2) if y1 <= y2 else (x2, y2, x1, y1)


def _line_angle_from_vertical(line):
    dx = abs(line[2] - line[0])
    dy = abs(line[3] - line[1])
    return float(np.degrees(np.arctan2(dx, dy + 1e-6)))


def _line_angle_deg(line):
    return float(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])))


def _line_intersection(l1, l2):
    """Find intersection of two infinite lines defined by endpoints."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def _angle_delta(a, b):
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _extend_line(line, y_top, y_bottom):
    """Extend a line to span from y_top to y_bottom."""
    xt = _line_x_at_y(line, y_top)
    xb = _line_x_at_y(line, y_bottom)
    return _clamp_line((
        xt if xt is not None else line[0], y_top,
        xb if xb is not None else line[2], y_bottom,
    ))


def _to_global(local, ox, oy):
    """Convert ROI-local coords to full-image coords."""
    return _clamp_line((local[0] + ox, local[1] + oy, local[2] + ox, local[3] + oy))


def _fit_centerline(left, right):
    if left is None or right is None:
        return None
    return (
        int((left[0] + right[0]) / 2), int((left[1] + right[1]) / 2),
        int((left[2] + right[2]) / 2), int((left[3] + right[3]) / 2),
    )


# ──────────────────────────────────────────────────────────────
# Temporal smoothing helpers
# ──────────────────────────────────────────────────────────────

def _smooth_val(cur, prev, alpha, max_delta):
    if prev is None:
        return cur
    s = (1.0 - alpha) * prev + alpha * cur
    d = s - prev
    if abs(d) > max_delta:
        s = prev + (max_delta if d > 0 else -max_delta)
    return float(s)


def _smooth_line(cur, prev, alpha, max_shift):
    if cur is None:
        return prev
    if prev is None:
        return cur
    return _clamp_line((
        _smooth_val(cur[0], prev[0], alpha, max_shift),
        _smooth_val(cur[1], prev[1], alpha, max_shift),
        _smooth_val(cur[2], prev[2], alpha, max_shift),
        _smooth_val(cur[3], prev[3], alpha, max_shift),
    ))


def _line_shift(cur, prev):
    """Total absolute pixel shift between two lines (sum of endpoint deltas)."""
    if cur is None or prev is None:
        return 0.0
    return max(
        abs(cur[0] - prev[0]) + abs(cur[2] - prev[2]),
        abs(cur[1] - prev[1]) + abs(cur[3] - prev[3]),
    )


# ──────────────────────────────────────────────────────────────
# ROI construction
# ──────────────────────────────────────────────────────────────

def _make_roi(det_box, w, h):
    """Build a tight ROI around the detection bbox. NO full-frame fallback."""
    x1, y1, x2, y2 = det_box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = max(8, int(ROI_PAD_X_FRAC * bw))
    pad_top = max(6, int(ROI_PAD_TOP_FRAC * bh))
    pad_bot = max(10, int(ROI_PAD_BOTTOM_FRAC * bh))
    return _safe_bbox((x1 - pad_x, y1 - pad_top, x2 + pad_x, y2 + pad_bot), w, h)


# ──────────────────────────────────────────────────────────────
# Preprocessing and corridor mask
# ──────────────────────────────────────────────────────────────

def _build_corridor_mask(roi_h, roi_w, det_box, roi_box, expected_half_width):
    """Build a trapezoidal mask matching expected runway corridor shape.
    
    Narrow at top (vanishing point direction), wider at bottom.
    """
    det_cx = (det_box[0] + det_box[2]) / 2.0
    cx_local = int(det_cx - roi_box[0])  # center in ROI coordinates
    top_half = max(10, int(0.50 * expected_half_width))
    bot_half = max(top_half + 8, int(1.60 * expected_half_width))
    y_top = max(0, int(0.02 * roi_h))
    y_bot = roi_h - 1
    pts = np.array([
        [max(0, cx_local - bot_half), y_bot],
        [max(0, cx_local - top_half), y_top],
        [min(roi_w - 1, cx_local + top_half), y_top],
        [min(roi_w - 1, cx_local + bot_half), y_bot],
    ], dtype=np.int32)
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _preprocess(roi, det_box, roi_box, expected_half_width):
    """Convert ROI to edge map, masked by the corridor."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    k = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    mask = _build_corridor_mask(
        roi.shape[0], roi.shape[1], det_box, roi_box, expected_half_width,
    )
    masked = cv2.bitwise_and(edges, edges, mask=mask)
    return masked, mask


# ──────────────────────────────────────────────────────────────
# Candidate filtering
# ──────────────────────────────────────────────────────────────

def _classify_candidates(raw_lines, roi_w, roi_h, expected_center_local, expected_half_width):
    """Filter raw Hough lines and classify as left or right candidates.
    
    Key perspective-awareness rules:
    - Left edges must lean right (dx > 0 from top to bottom) toward vanishing point
    - Right edges must lean left (dx < 0 from top to bottom) toward vanishing point
    - Both must be roughly near the expected center
    """
    left = []
    right = []
    reject = {
        "too_short": 0, "bad_angle": 0, "short_span": 0,
        "outside_corridor": 0, "wrong_lean": 0,
    }
    if raw_lines is None:
        return left, right, reject

    lat_tol = max(12.0, LATERAL_TOLERANCE_FRAC * roi_w, 1.1 * expected_half_width)

    for raw in raw_lines:
        line = _line_top_bottom(_clamp_line(raw[0]))
        x1, y1, x2, y2 = line

        # Length filter
        if _line_length(line) < MIN_LINE_LENGTH:
            reject["too_short"] += 1
            continue

        # Angle filter
        angle = _line_angle_from_vertical(line)
        if angle < MIN_ANGLE_FROM_VERTICAL or angle > MAX_ANGLE_FROM_VERTICAL:
            reject["bad_angle"] += 1
            continue

        # Vertical span filter
        v_span = abs(y2 - y1)
        if v_span < MIN_VERTICAL_SPAN_FRAC * roi_h:
            reject["short_span"] += 1
            continue

        # Lateral position filter
        mid_y = (y1 + y2) / 2.0
        x_mid = _line_x_at_y(line, mid_y)
        if x_mid is None:
            reject["short_span"] += 1
            continue
        if abs(x_mid - expected_center_local) > lat_tol:
            reject["outside_corridor"] += 1
            continue

        # Perspective lean filter: runway edges converge toward vanishing point
        # dx = x_bottom - x_top (since y2 >= y1 after _line_top_bottom)
        dx = x2 - x1  # positive = line leans right going down

        if x_mid <= expected_center_local:
            # Left edge: must lean toward center (rightward), so dx > 0
            if dx < 1:
                reject["wrong_lean"] += 1
                continue
            left.append(line)
        else:
            # Right edge: must lean toward center (leftward), so dx < 0
            if dx > -1:
                reject["wrong_lean"] += 1
                continue
            right.append(line)

    # Sort by length (longest first) for greedy pairing
    left.sort(key=_line_length, reverse=True)
    right.sort(key=_line_length, reverse=True)
    return left, right, reject


# ──────────────────────────────────────────────────────────────
# Pair scoring
# ──────────────────────────────────────────────────────────────

def _score_pair(left, right, roi_w, roi_h, expected_center, expected_half_width, prev_center):
    """Score a left+right line pair for how well they describe a runway corridor."""
    y_top = int(0.20 * roi_h)
    y_bot = int(0.90 * roi_h)

    lx_t = _line_x_at_y(left, y_top)
    rx_t = _line_x_at_y(right, y_top)
    lx_b = _line_x_at_y(left, y_bot)
    rx_b = _line_x_at_y(right, y_bot)
    if None in (lx_t, rx_t, lx_b, rx_b):
        return -1.0

    width_top = rx_t - lx_t
    width_bot = rx_b - lx_b

    # Basic sanity: right must be right of left, wider at bottom than top
    if width_top <= 0 or width_bot <= 0:
        return -1.0
    if width_bot <= width_top:
        return -1.0  # runway must widen toward camera
    if width_bot > 0.85 * roi_w:
        return -1.0  # too wide = probably background
    if width_top < 0.01 * roi_w:
        return -1.0  # too narrow at top = degenerate

    # Vanishing point check
    vp = _line_intersection(left, right)
    if vp is not None:
        vx, vy = vp
        center_dist = abs(vx - expected_center) / max(1.0, roi_w)
        if center_dist > VP_X_TOLERANCE:
            return -1.0
        if vy > VP_Y_MAX * roi_h:
            return -1.0  # VP should be above the ROI or near top

    # Score components
    expected_w = 2.0 * expected_half_width

    # Width match term
    w_term = 1.0 - min(1.0, abs(width_bot - expected_w) / max(1.0, 0.5 * roi_w))

    # Center position term (weighted top + bottom)
    c_bot = (lx_b + rx_b) / 2.0
    c_top = (lx_t + rx_t) / 2.0
    c_term = 0.5 * (1.0 - min(1.0, abs(c_bot - expected_center) / max(1.0, 0.3 * roi_w)))
    c_term += 0.5 * (1.0 - min(1.0, abs(c_top - expected_center) / max(1.0, 0.25 * roi_w)))

    # Perspective spread term (wider at bottom)
    spread_term = min(1.0, (width_bot - width_top) / max(1.0, 0.15 * roi_w))

    # Length term
    len_term = min(1.0, (_line_length(left) + _line_length(right)) / max(1.0, 1.4 * roi_h))

    # Temporal continuity term
    if prev_center is not None:
        prev_term = 1.0 - min(1.0, abs(c_bot - prev_center) / max(1.0, 0.3 * roi_w))
    else:
        prev_term = 0.5

    # Symmetry term
    symmetry = 1.0 - min(1.0, abs(
        abs(c_bot - lx_b) - abs(rx_b - c_bot)
    ) / max(1.0, 0.5 * width_bot))

    return (0.25 * w_term + 0.22 * c_term + 0.15 * spread_term +
            0.13 * len_term + 0.13 * prev_term + 0.12 * symmetry)


def _find_best_pair(left_lines, right_lines, roi_w, roi_h,
                    expected_center, expected_half_width, prev_center):
    """Try all left-right combinations and return the highest-scoring pair."""
    if not left_lines or not right_lines:
        return None, None, -1.0, "no_candidates"

    best_l = None
    best_r = None
    best_s = -1.0

    for l in left_lines[:30]:
        for r in right_lines[:30]:
            # Quick check: right must be right of left at midpoint
            mid_y = int(0.6 * roi_h)
            lx = _line_x_at_y(l, mid_y)
            rx = _line_x_at_y(r, mid_y)
            if lx is None or rx is None or rx <= lx:
                continue
            s = _score_pair(l, r, roi_w, roi_h, expected_center, expected_half_width, prev_center)
            if s > best_s:
                best_s = s
                best_l, best_r = l, r

    if best_l is None or best_s < MIN_PAIR_SCORE:
        return None, None, best_s, "score_below_threshold"
    return best_l, best_r, best_s, "accepted"


# ──────────────────────────────────────────────────────────────
# Bbox corridor fallback
# ──────────────────────────────────────────────────────────────

def _bbox_corridor(det_box, w, h, prev_state):
    """Derive a perspective-correct runway corridor directly from the YOLO bbox.
    
    This is used when edge detection fails. It produces a believable
    corridor that is narrower at top and wider at bottom.
    """
    x1, y1, x2, y2 = det_box
    bw = max(1, x2 - x1)
    center_x = int((x1 + x2) / 2)
    half_w = max(6, int(0.30 * bw))

    # Blend with previous state for stability
    if prev_state:
        prev_half = prev_state.get("half_width")
        prev_cx = prev_state.get("center_bottom_x")
        if prev_half is not None and prev_cx is not None:
            half_w = int(0.6 * half_w + 0.4 * float(prev_half))
            center_x = int(0.6 * center_x + 0.4 * float(prev_cx))

    band_top = max(0, y1 - int(0.05 * max(1, y2 - y1)))
    band_bot = min(h - 1, y2 + int(0.10 * max(1, y2 - y1)))
    top_half = max(4, int(0.55 * half_w))  # narrower at top

    left_line = (center_x - top_half, band_top, center_x - half_w, band_bot)
    right_line = (center_x + top_half, band_top, center_x + half_w, band_bot)
    center_line = (center_x, band_top, center_x, band_bot)
    return left_line, right_line, center_line, float(half_w)


# ──────────────────────────────────────────────────────────────
# Drawing / overlay
# ──────────────────────────────────────────────────────────────

def _draw_overlay(image, left, right, center, image_center_x, band, source, offset_px):
    """Draw geometry overlay on the annotated image."""
    band_top = max(0, band[0])
    band_bot = min(image.shape[0] - 1, band[1])

    def clamp(line):
        return _extend_line(line, band_top, band_bot)

    if left is not None:
        l = clamp(left)
        cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    if right is not None:
        r = clamp(right)
        cv2.line(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 3, cv2.LINE_AA)
    if center is not None:
        c = clamp(center)
        cv2.line(image, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 2, cv2.LINE_AA)

    # HUD text
    cv2.putText(image, "geo: " + source, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if offset_px is not None:
        cv2.putText(image, "offset: " + str(round(offset_px, 1)) + "px", (16, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2, cv2.LINE_AA)
    return image


def _debug_lines_image(roi_image, raw_lines, sel_left, sel_right):
    """Produce a debug visualization of raw + selected lines in ROI space."""
    vis = roi_image.copy()
    if raw_lines is not None:
        for raw in raw_lines:
            x1, y1, x2, y2 = _clamp_line(raw[0])
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    if sel_left is not None:
        cv2.line(vis, (sel_left[0], sel_left[1]), (sel_left[2], sel_left[3]), (0, 0, 255), 3)
    if sel_right is not None:
        cv2.line(vis, (sel_right[0], sel_right[1]), (sel_right[2], sel_right[3]), (255, 255, 0), 3)
    return vis


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def estimate_runway_geometry(image, detection, prev_state=None):
    """Estimate left/right runway edges from a single frame.
    
    Args:
        image: BGR numpy array (full frame)
        detection: dict with "bbox" key from YOLO detector
        prev_state: tracker_state dict from previous frame (for temporal smoothing)
    
    Returns:
        dict with geometry results, debug info, annotated image, and tracker_state
    """
    h, w = image.shape[:2]
    image_center_x = w // 2
    prev_state = prev_state or {}

    # ── No detection → empty result ──
    if not detection.get("bbox"):
        logger.debug("No YOLO detection - returning empty geometry")
        return _empty_result(image, image_center_x)

    det_box = _safe_bbox(detection["bbox"], w, h)
    det_cx = (det_box[0] + det_box[2]) / 2.0
    det_w = max(1, det_box[2] - det_box[0])

    # ── Build expectations from detection + prior state ──
    prev_cx = prev_state.get("center_bottom_x")
    prev_half = prev_state.get("half_width")
    prev_score = prev_state.get("pair_score", 0.0)

    # Blend current detection center with prior for stability
    if prev_cx is not None and prev_score >= REUSE_PREV_SCORE_REQ:
        expected_center = 0.5 * float(det_cx) + 0.5 * float(prev_cx)
    else:
        expected_center = float(det_cx)

    expected_half_w = float(0.35 * det_w)
    if prev_half is not None:
        expected_half_w = 0.6 * expected_half_w + 0.4 * float(prev_half)

    # ── Build tight ROI ──
    roi_box = _make_roi(det_box, w, h)
    rx1, ry1, rx2, ry2 = roi_box
    roi_image = image[ry1:ry2, rx1:rx2]

    debug = GeometryDebug(roi_source="bbox_tight", roi_box=[rx1, ry1, rx2, ry2])

    if roi_image.size == 0:
        logger.warning("Empty ROI - using bbox fallback")
        return _fallback_result(image, det_box, w, h, image_center_x, prev_state, debug)

    roi_h, roi_w = roi_image.shape[:2]

    # ── Edge detection with corridor mask ──
    edges, corridor_mask = _preprocess(roi_image, det_box, roi_box, expected_half_w)

    raw_lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=max(20, int(HOUGH_MIN_LENGTH_FRAC * roi_h)),
        maxLineGap=HOUGH_MAX_GAP,
    )
    debug.raw_line_count = 0 if raw_lines is None else int(len(raw_lines))

    # ── Classify candidates ──
    center_local = expected_center - rx1
    left_cands, right_cands, rejects = _classify_candidates(
        raw_lines, roi_w, roi_h, center_local, expected_half_w,
    )
    debug.left_candidate_count = len(left_cands)
    debug.right_candidate_count = len(right_cands)
    debug.candidate_reject_count = rejects

    # ── Find best pair ──
    prev_center_local = None if prev_cx is None else prev_cx - rx1
    pair_left, pair_right, pair_score, pair_reason = _find_best_pair(
        left_cands, right_cands, roi_w, roi_h,
        center_local, expected_half_w, prev_center_local,
    )
    debug.pair_score = pair_score
    debug.pairing_reason = pair_reason

    # ── Debug visualizations ──
    lines_image = _debug_lines_image(roi_image, raw_lines, pair_left, pair_right)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    logger.info(
        "geometry: raw=%d L=%d R=%d rejects=%s pair=%.3f reason=%s",
        debug.raw_line_count, len(left_cands), len(right_cands),
        rejects, pair_score, pair_reason,
    )

    # ── Apply temporal smoothing / decide geometry source ──
    geometry_source = "bbox_fallback"
    smoothed = False
    reused = False
    selected_left = None
    selected_right = None
    centerline = None
    runway_cx_bottom = None
    offset_px = None

    band_top = max(0, det_box[1] - int(0.05 * max(1, det_box[3] - det_box[1])))
    band_bot = min(h - 1, det_box[3] + int(0.10 * max(1, det_box[3] - det_box[1])))
    runway_band = (band_top, band_bot)

    max_shift = max(10, int(MAX_SHIFT_FRAC * w))
    jump_threshold = max(20.0, JUMP_THRESHOLD_FRAC * w)

    if pair_left is not None and pair_right is not None and pair_score >= MIN_PAIR_SCORE:
        # Convert to global coords and extend to full band
        gl = _to_global(_extend_line(pair_left, 0, roi_h - 1), rx1, ry1)
        gr = _to_global(_extend_line(pair_right, 0, roi_h - 1), rx1, ry1)

        prev_left = prev_state.get("left_edge")
        prev_right = prev_state.get("right_edge")

        # Smooth against previous frame
        sm_left = _smooth_line(gl, prev_left, SMOOTH_ALPHA, max_shift)
        sm_right = _smooth_line(gr, prev_right, SMOOTH_ALPHA, max_shift)

        # Check for jumps
        jump_l = _line_shift(sm_left, prev_left)
        jump_r = _line_shift(sm_right, prev_right)
        angle_jump = 0.0
        if sm_left and sm_right and prev_left and prev_right:
            angle_jump = max(
                _angle_delta(_line_angle_deg(sm_left), _line_angle_deg(prev_left)),
                _angle_delta(_line_angle_deg(sm_right), _line_angle_deg(prev_right)),
            )

        if (prev_score >= REUSE_PREV_SCORE_REQ
                and prev_left is not None and prev_right is not None
                and (jump_l > jump_threshold or jump_r > jump_threshold
                     or angle_jump > ANGLE_JUMP_MAX)):
            logger.info(
                "Rejecting edge pair (jump L=%.1f R=%.1f angle=%.1f) - reusing previous",
                jump_l, jump_r, angle_jump,
            )
            selected_left, selected_right = prev_left, prev_right
            geometry_source = "reused_prev"
            reused = True
            debug.reused_geometry = True
        else:
            selected_left = sm_left
            selected_right = sm_right
            geometry_source = "edges"
            if sm_left != gl or sm_right != gr:
                smoothed = True
                debug.smoothed_from_previous = True

    elif (prev_score >= REUSE_PREV_SCORE_REQ
          and prev_state.get("left_edge") and prev_state.get("right_edge")):
        # No good pair found this frame but previous was good → reuse
        logger.info(
            "No valid edge pair (score=%.3f) - reusing previous geometry", pair_score,
        )
        selected_left = prev_state["left_edge"]
        selected_right = prev_state["right_edge"]
        geometry_source = "reused_prev"
        reused = True
        debug.reused_geometry = True

    # ── Compute centerline + offset, or fall back to bbox corridor ──
    if selected_left is not None and selected_right is not None:
        selected_left = _extend_line(selected_left, band_top, band_bot)
        selected_right = _extend_line(selected_right, band_top, band_bot)
        centerline = _fit_centerline(selected_left, selected_right)
        if centerline is not None:
            runway_cx_bottom = centerline[2]
            offset_px = float(runway_cx_bottom - image_center_x)
    else:
        logger.info("Using bbox corridor fallback")
        fb_left, fb_right, fb_center, fb_half = _bbox_corridor(det_box, w, h, prev_state)
        selected_left, selected_right, centerline = fb_left, fb_right, fb_center
        runway_cx_bottom = fb_center[2]
        offset_px = float(runway_cx_bottom - image_center_x)
        geometry_source = "bbox_fallback"

    debug.selected_left = selected_left
    debug.selected_right = selected_right

    if selected_left is not None and selected_right is not None:
        half_w_out = max(6.0, abs(selected_right[2] - selected_left[2]) / 2.0)
    else:
        half_w_out = expected_half_w

    # Glidepath line from image center bottom to runway center top
    glidepath_line = None
    if centerline is not None:
        glidepath_line = (image_center_x, band_bot, centerline[2], centerline[1])

    # ── Build tracker state for next frame ──
    tracker_state = {
        "left_edge": selected_left,
        "right_edge": selected_right,
        "centerline": centerline,
        "center_bottom_x": runway_cx_bottom,
        "half_width": half_w_out,
        "pair_score": pair_score if geometry_source == "edges" else (
            prev_score * 0.9 if reused else 0.0
        ),
        "source": geometry_source,
    }

    return {
        "left_edge": selected_left,
        "right_edge": selected_right,
        "centerline": centerline,
        "glidepath_line": glidepath_line,
        "runway_center_bottom_x": runway_cx_bottom,
        "image_center_x": image_center_x,
        "offset_px": offset_px,
        "geometry_source": geometry_source,
        "roi_box": debug.roi_box,
        "roi_image": roi_image,
        "edges_image": edges_bgr,
        "lines_image": lines_image,
        "pair_score": pair_score if pair_score is not None else 0.0,
        "roi_source": debug.roi_source,
        "smoothed": smoothed,
        "reused_from_previous": reused,
        "candidate_reject_count": debug.candidate_reject_count,
        "left_candidate_count": debug.left_candidate_count,
        "right_candidate_count": debug.right_candidate_count,
        "raw_line_count": debug.raw_line_count,
        "pairing_reason": debug.pairing_reason,
        "geometry_debug": debug.__dict__,
        "tracker_state": tracker_state,
        "annotated_image": image,
    }


# ──────────────────────────────────────────────────────────────
# Result builders for empty / fallback cases
# ──────────────────────────────────────────────────────────────

def _empty_result(frame, image_center_x):
    """Return structure when no detection is available."""
    return {
        "left_edge": None, "right_edge": None, "centerline": None,
        "glidepath_line": None, "runway_center_bottom_x": None,
        "image_center_x": image_center_x, "offset_px": None,
        "geometry_source": "none", "roi_box": None,
        "roi_image": None, "edges_image": None, "lines_image": None,
        "pair_score": 0.0, "roi_source": "none",
        "smoothed": False, "reused_from_previous": False,
        "candidate_reject_count": {}, "left_candidate_count": 0,
        "right_candidate_count": 0, "raw_line_count": 0,
        "pairing_reason": "no_detection",
        "geometry_debug": GeometryDebug().__dict__,
        "tracker_state": {
            "left_edge": None, "right_edge": None, "centerline": None,
            "center_bottom_x": None, "half_width": None,
            "pair_score": 0.0, "source": "none",
        },
        "annotated_image": frame,
    }


def _fallback_result(frame, det_box, w, h, image_center_x, prev_state, debug):
    """Return structure using bbox-derived corridor when edge detection fails entirely."""
    fb_left, fb_right, fb_center, fb_half = _bbox_corridor(det_box, w, h, prev_state)
    offset_px = float(fb_center[2] - image_center_x)
    debug.pairing_reason = "bbox_fallback"
    return {
        "left_edge": fb_left, "right_edge": fb_right, "centerline": fb_center,
        "glidepath_line": (image_center_x, band_bot, fb_center[2], fb_center[1]),
        "runway_center_bottom_x": fb_center[2],
        "image_center_x": image_center_x, "offset_px": offset_px,
        "geometry_source": "bbox_fallback", "roi_box": debug.roi_box,
        "roi_image": None, "edges_image": None, "lines_image": None,
        "pair_score": 0.0, "roi_source": debug.roi_source,
        "smoothed": False, "reused_from_previous": False,
        "candidate_reject_count": {}, "left_candidate_count": 0,
        "right_candidate_count": 0, "raw_line_count": 0,
        "pairing_reason": "bbox_fallback",
        "geometry_debug": debug.__dict__,
        "tracker_state": {
            "left_edge": fb_left, "right_edge": fb_right, "centerline": fb_center,
            "center_bottom_x": fb_center[2], "half_width": fb_half,
            "pair_score": 0.0, "source": "bbox_fallback",
        },
        "annotated_image": frame,
    }
