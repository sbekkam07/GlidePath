"""Runway geometry extraction and polished runway overlay rendering."""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

Line = Tuple[int, int, int, int]
Geometry = Dict[str, Any]

LEFT_CLASS = "left"
RIGHT_CLASS = "right"

# Geometry extraction tuning.
MIN_VERTICAL_COVERAGE_RATIO = 0.22
MIN_LINE_LENGTH_RATIO = 0.30
MIN_SLOPE_ABS = 0.01
MAX_SLOPE_ABS = 1.20
MIN_LINE_GAP = 20

# Overlay visual style.
RUNWAY_BOX_COLOR = (0, 190, 255)  # BGR orange
RUNWAY_FILL_COLOR = (0, 90, 255)  # BGR orange-red
LEFT_EDGE_COLOR = (0, 120, 255)
RIGHT_EDGE_COLOR = (0, 170, 255)
CENTERLINE_COLOR = (255, 255, 255)
LABEL_TEXT_COLOR = (255, 255, 255)
GUIDANCE_TEXT_COLOR = (255, 255, 255)
GUIDANCE_ALIGNED_BG = (46, 160, 67)
GUIDANCE_CORRECTION_BG = (0, 120, 255)

BOX_THICKNESS = 2
GEOMETRY_THICKNESS = 2
RUNWAY_FILL_ALPHA = 0.24
GEOMETRY_DRAW_THRESHOLD = 0.58
GUIDANCE_MIN_THRESHOLD_PX = 10.0
GUIDANCE_THRESHOLD_RATIO = 0.01


def _empty_geometry(image_width: int, bbox: list[int] | None = None) -> Geometry:
    return {
        "bbox": list(bbox) if bbox else None,
        "left_edge": None,
        "right_edge": None,
        "centerline": None,
        "center_x_bottom": None,
        "image_center_x": image_width // 2,
        "signed_offset_px": None,
        "runway_polygon": None,
        "geometry_confidence": 0.0,
    }


def _line_x_at_y(line: Line, y: int) -> float:
    x1, y1, x2, y2 = line
    if y2 == y1:
        return float(x1)
    t = (y - y1) / float(y2 - y1)
    return x1 + t * (x2 - x1)


def _line_length(line: Line) -> float:
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def _ensure_ordered_top_bottom(line: Line) -> Line:
    x1, y1, x2, y2 = line
    if y1 <= y2:
        return (x1, y1, x2, y2)
    return (x2, y2, x1, y1)


def _clip(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _is_valid_line_list(line: Any) -> bool:
    return isinstance(line, list) and len(line) == 4


def _to_line_tuple(line: Any) -> Line | None:
    if not _is_valid_line_list(line):
        return None
    return tuple(int(v) for v in line)  # type: ignore[return-value]


def _clip_line_to_frame(line: Line, frame_shape: tuple[int, int, int]) -> Line:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = line
    return (
        _clip(int(round(x1)), 0, w - 1),
        _clip(int(round(y1)), 0, h - 1),
        _clip(int(round(x2)), 0, w - 1),
        _clip(int(round(y2)), 0, h - 1),
    )


def _extrapolate_line_to_crop_bounds(line: Line, crop_h: int, crop_w: int) -> Optional[Line]:
    x1, y1, x2, y2 = _ensure_ordered_top_bottom(line)
    if y1 == y2 or crop_h <= 1 or crop_w <= 1:
        return None

    y_top = 0
    y_bottom = crop_h - 1
    top_x = int(round(_clip(int(round(_line_x_at_y((x1, y1, x2, y2), y_top))), 0, crop_w - 1)))
    bottom_x = int(round(_clip(int(round(_line_x_at_y((x1, y1, x2, y2), y_bottom))), 0, crop_w - 1)))
    return (top_x, y_top, bottom_x, y_bottom)


def _classify_and_score_candidate(
    line: Line,
    crop_w: int,
    crop_h: int,
) -> tuple[str, float, Line, float, float] | None:
    """Classify a Hough line into left/right boundary with a quality score."""
    x_top, y_top, x_bottom, y_bottom = _ensure_ordered_top_bottom(line)
    dy = y_bottom - y_top
    dx = x_bottom - x_top
    if dy <= max(8, int(crop_h * MIN_VERTICAL_COVERAGE_RATIO)):
        return None

    length = _line_length((x_top, y_top, x_bottom, y_bottom))
    if length < max(28.0, float(crop_h) * MIN_LINE_LENGTH_RATIO):
        return None

    slope = dx / float(max(1, dy))
    slope_abs = abs(slope)
    if slope_abs < MIN_SLOPE_ABS or slope_abs > MAX_SLOPE_ABS:
        return None

    bottom_x_norm = x_bottom / float(max(1, crop_w - 1))
    coverage = min(1.0, dy / float(max(1, crop_h)))
    verticality = min(1.0, abs(dy) / (abs(dx) + 1e-6))

    if slope < 0 and bottom_x_norm < 0.76:
        side = LEFT_CLASS
        target_x = 0.24
    elif slope > 0 and bottom_x_norm > 0.24:
        side = RIGHT_CLASS
        target_x = 0.76
    else:
        return None

    position_score = max(0.1, 1.0 - abs(bottom_x_norm - target_x) / 0.70)
    score = length * (0.45 + 0.55 * verticality) * (0.5 + 0.5 * coverage) * position_score
    return side, float(score), (x_top, y_top, x_bottom, y_bottom), float(slope), float(coverage)


def _build_runway_polygon(
    left_edge: list[int] | None,
    right_edge: list[int] | None,
    frame_shape: tuple[int, int, int],
) -> list[list[int]] | None:
    left_line = _to_line_tuple(left_edge)
    right_line = _to_line_tuple(right_edge)
    if left_line is None or right_line is None:
        return None

    left_line = _clip_line_to_frame(left_line, frame_shape)
    right_line = _clip_line_to_frame(right_line, frame_shape)
    left_top_x, left_top_y, left_bottom_x, left_bottom_y = left_line
    right_top_x, right_top_y, right_bottom_x, right_bottom_y = right_line

    polygon = np.array(
        [
            [left_bottom_x, left_bottom_y],
            [right_bottom_x, right_bottom_y],
            [right_top_x, right_top_y],
            [left_top_x, left_top_y],
        ],
        dtype=np.int32,
    )
    area = float(cv2.contourArea(polygon.astype(np.float32)))
    if area < 120.0:
        return None

    return polygon.tolist()


def _draw_compact_bbox_label(
    frame: np.ndarray,
    bbox: list[int],
    detector_confidence: Optional[float],
) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = _clip(x1, 0, max(0, w - 1))
    x2 = _clip(x2, 0, max(0, w - 1))
    y1 = _clip(y1, 0, max(0, h - 1))
    y2 = _clip(y2, 0, max(0, h - 1))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        RUNWAY_BOX_COLOR,
        BOX_THICKNESS,
        cv2.LINE_AA,
    )

    conf = float(detector_confidence) if isinstance(detector_confidence, (int, float)) else 0.0
    label = f"runway {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad_x = 6
    pad_y = 3

    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2
    label_x = _clip(x1, 0, max(0, w - box_w))
    label_y_bottom = y1 - 5
    if label_y_bottom - box_h < 0:
        label_y_bottom = _clip(y1 + box_h + 2, box_h, h - 1)

    top_left = (label_x, label_y_bottom - box_h)
    bottom_right = (label_x + box_w, label_y_bottom)
    cv2.rectangle(frame, top_left, bottom_right, RUNWAY_BOX_COLOR, -1, cv2.LINE_AA)

    text_org = (top_left[0] + pad_x, bottom_right[1] - baseline - pad_y)
    cv2.putText(
        frame,
        label,
        text_org,
        font,
        font_scale,
        LABEL_TEXT_COLOR,
        text_thickness,
        cv2.LINE_AA,
    )


def _geometry_draw_allowed(geometry: Geometry) -> bool:
    if not isinstance(geometry, dict):
        return False
    if not _is_valid_line_list(geometry.get("left_edge")):
        return False
    if not _is_valid_line_list(geometry.get("right_edge")):
        return False
    return float(geometry.get("geometry_confidence", 0.0) or 0.0) >= GEOMETRY_DRAW_THRESHOLD


def get_guidance_label(
    signed_offset_px: Any,
    frame_width: int,
) -> tuple[str, tuple[int, int, int]]:
    threshold = max(GUIDANCE_MIN_THRESHOLD_PX, float(frame_width) * GUIDANCE_THRESHOLD_RATIO)
    if not isinstance(signed_offset_px, (int, float)):
        return "ALIGNED", GUIDANCE_ALIGNED_BG
    offset_value = float(signed_offset_px)
    if offset_value > threshold:
        return "CORRECT RIGHT", GUIDANCE_CORRECTION_BG
    if offset_value < -threshold:
        return "CORRECT LEFT", GUIDANCE_CORRECTION_BG
    return "ALIGNED", GUIDANCE_ALIGNED_BG


def _draw_guidance_box(
    frame: np.ndarray,
    signed_offset_px: Any,
) -> None:
    h, w = frame.shape[:2]
    label, background_color = get_guidance_label(signed_offset_px, w)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    text_thickness = 1
    pad_x = 10
    pad_y = 6
    margin = 14

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    box_w = text_w + (pad_x * 2)
    box_h = text_h + baseline + (pad_y * 2)
    box_x1 = max(0, w - box_w - margin)
    box_y1 = max(0, h - box_h - margin)
    box_x2 = min(w - 1, box_x1 + box_w)
    box_y2 = min(h - 1, box_y1 + box_h)

    cv2.rectangle(
        frame,
        (box_x1, box_y1),
        (box_x2, box_y2),
        background_color,
        -1,
        cv2.LINE_AA,
    )
    text_org = (box_x1 + pad_x, box_y2 - baseline - pad_y)
    cv2.putText(
        frame,
        label,
        text_org,
        font,
        font_scale,
        GUIDANCE_TEXT_COLOR,
        text_thickness,
        cv2.LINE_AA,
    )


def draw_runway_overlay(
    frame: np.ndarray,
    geometry: Geometry,
    frame_index: int | None = None,
    alignment: Optional[str] = None,
    detector_confidence: Optional[float] = None,
) -> np.ndarray:
    """Draw runway bbox by default; only draw extra geometry when confidence is high."""
    del frame_index, alignment  # Kept for API compatibility.

    if frame is None:
        return frame
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    rendered = frame.copy()
    h, w = rendered.shape[:2]

    if _geometry_draw_allowed(geometry):
        polygon = geometry.get("runway_polygon")
        if not isinstance(polygon, list):
            polygon = _build_runway_polygon(
                geometry.get("left_edge"),
                geometry.get("right_edge"),
                rendered.shape,
            )
        if isinstance(polygon, list) and len(polygon) == 4:
            polygon_np = np.array(
                [[_clip(int(pt[0]), 0, w - 1), _clip(int(pt[1]), 0, h - 1)] for pt in polygon],
                dtype=np.int32,
            )
            color_overlay = rendered.copy()
            cv2.fillPoly(color_overlay, [polygon_np], RUNWAY_FILL_COLOR, cv2.LINE_AA)
            rendered = cv2.addWeighted(
                color_overlay,
                RUNWAY_FILL_ALPHA,
                rendered,
                1.0 - RUNWAY_FILL_ALPHA,
                0.0,
            )

        left = _to_line_tuple(geometry.get("left_edge"))
        right = _to_line_tuple(geometry.get("right_edge"))
        if left is not None:
            left = _clip_line_to_frame(left, rendered.shape)
            cv2.line(
                rendered,
                (left[0], left[1]),
                (left[2], left[3]),
                LEFT_EDGE_COLOR,
                GEOMETRY_THICKNESS,
                cv2.LINE_AA,
            )
        if right is not None:
            right = _clip_line_to_frame(right, rendered.shape)
            cv2.line(
                rendered,
                (right[0], right[1]),
                (right[2], right[3]),
                RIGHT_EDGE_COLOR,
                GEOMETRY_THICKNESS,
                cv2.LINE_AA,
            )

        centerline = _to_line_tuple(geometry.get("centerline"))
        if centerline is not None:
            centerline = _clip_line_to_frame(centerline, rendered.shape)
            cv2.line(
                rendered,
                (centerline[0], centerline[1]),
                (centerline[2], centerline[3]),
                CENTERLINE_COLOR,
                GEOMETRY_THICKNESS,
                cv2.LINE_AA,
            )

    bbox = geometry.get("bbox")
    if _is_valid_line_list(bbox):
        _draw_compact_bbox_label(rendered, [int(v) for v in bbox], detector_confidence)

    _draw_guidance_box(rendered, geometry.get("signed_offset_px"))

    return rendered


def estimate_runway_geometry(
    image: np.ndarray,
    detection: Dict[str, Any],
) -> Geometry:
    """Estimate runway edges and centerline inside a YOLO runway ROI."""
    if image is None or image.size == 0:
        return _empty_geometry(0)
    if detection is None or detection.get("bbox") is None:
        return _empty_geometry(image.shape[1])

    h, w = image.shape[:2]
    geometry = _empty_geometry(w, detection.get("bbox"))
    image_center_x = w // 2
    geometry["image_center_x"] = image_center_x

    raw_bbox = detection.get("bbox")
    if not _is_valid_line_list(raw_bbox):
        return geometry

    x1, y1, x2, y2 = [int(v) for v in raw_bbox]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = _clip(x1, 0, w - 1)
    x2 = _clip(x2, 0, w - 1)
    y1 = _clip(y1, 0, h - 1)
    y2 = _clip(y2, 0, h - 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return geometry

    # Expand slightly to capture full runway boundaries at bbox edges.
    pad_x = max(2, int((x2 - x1) * 0.05))
    pad_y = max(2, int((y2 - y1) * 0.05))
    x1 = _clip(x1 - pad_x, 0, w - 1)
    x2 = _clip(x2 + pad_x, 0, w - 1)
    y1 = _clip(y1 - int(pad_y * 0.5), 0, h - 1)
    y2 = _clip(y2 + pad_y, 0, h - 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return geometry

    geometry["bbox"] = [x1, y1, x2, y2]
    crop = image[y1 : y2 + 1, x1 : x2 + 1]
    if crop.size == 0:
        return geometry

    crop_h, crop_w = crop.shape[:2]
    if crop_h < 20 or crop_w < 20:
        return geometry

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_intensity = float(np.median(blurred))
    canny_low = max(35, int(0.66 * median_intensity))
    canny_high = min(220, max(canny_low + 30, int(1.33 * median_intensity)))
    edges = cv2.Canny(blurred, canny_low, canny_high)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(min(crop_w, crop_h) * 0.18)),
        minLineLength=max(30, int(crop_h * 0.35)),
        maxLineGap=max(MIN_LINE_GAP, int(crop_h * 0.08)),
    )
    if raw_lines is None:
        geometry["geometry_confidence"] = 0.0
        return geometry

    left_candidates: list[dict[str, float | Line]] = []
    right_candidates: list[dict[str, float | Line]] = []
    for hough_line in raw_lines[:, 0]:
        candidate = _classify_and_score_candidate(
            (int(hough_line[0]), int(hough_line[1]), int(hough_line[2]), int(hough_line[3])),
            crop_w,
            crop_h,
        )
        if candidate is None:
            continue
        side, score, line, slope, coverage = candidate
        packed = {
            "score": float(score),
            "line": line,
            "slope": float(slope),
            "coverage": float(coverage),
        }
        if side == LEFT_CLASS:
            left_candidates.append(packed)
        elif side == RIGHT_CLASS:
            right_candidates.append(packed)

    left_best = max(left_candidates, key=lambda item: float(item["score"])) if left_candidates else None
    right_best = max(right_candidates, key=lambda item: float(item["score"])) if right_candidates else None

    if left_best is not None:
        left_crop = _extrapolate_line_to_crop_bounds(left_best["line"], crop_h, crop_w)  # type: ignore[arg-type]
        if left_crop is not None:
            geometry["left_edge"] = [left_crop[0] + x1, left_crop[1] + y1, left_crop[2] + x1, left_crop[3] + y1]
    if right_best is not None:
        right_crop = _extrapolate_line_to_crop_bounds(right_best["line"], crop_h, crop_w)  # type: ignore[arg-type]
        if right_crop is not None:
            geometry["right_edge"] = [right_crop[0] + x1, right_crop[1] + y1, right_crop[2] + x1, right_crop[3] + y1]

    left_edge = geometry.get("left_edge")
    right_edge = geometry.get("right_edge")
    top_width = 0.0
    bottom_width = 0.0
    if _is_valid_line_list(left_edge) and _is_valid_line_list(right_edge):
        left_top_x, left_top_y, left_bottom_x, left_bottom_y = [int(v) for v in left_edge]
        right_top_x, right_top_y, right_bottom_x, right_bottom_y = [int(v) for v in right_edge]
        top_width = float(right_top_x - left_top_x)
        bottom_width = float(right_bottom_x - left_bottom_x)

        if top_width > 2 and bottom_width > 4:
            center_top_x = mean([left_top_x, right_top_x])
            center_top_y = mean([left_top_y, right_top_y])
            center_bottom_x = mean([left_bottom_x, right_bottom_x])
            center_bottom_y = mean([left_bottom_y, right_bottom_y])
            geometry["centerline"] = [
                int(round(center_top_x)),
                int(round(center_top_y)),
                int(round(center_bottom_x)),
                int(round(center_bottom_y)),
            ]
            geometry["center_x_bottom"] = float(center_bottom_x)
            geometry["signed_offset_px"] = float(center_bottom_x - image_center_x)

    geometry["runway_polygon"] = _build_runway_polygon(
        geometry.get("left_edge"),
        geometry.get("right_edge"),
        image.shape,
    )

    detector_conf = float(detection.get("confidence", 0.0) or 0.0)
    if left_best is not None and right_best is not None:
        left_cov = float(left_best["coverage"])
        right_cov = float(right_best["coverage"])
        coverage_score = min(1.0, (left_cov + right_cov) / 1.6)

        left_slope_abs = abs(float(left_best["slope"]))
        right_slope_abs = abs(float(right_best["slope"]))
        parallel_error = abs(left_slope_abs - right_slope_abs)
        parallel_score = max(0.0, 1.0 - (parallel_error / 0.55))

        width_bottom_norm = bottom_width / float(max(1, crop_w))
        width_top_norm = top_width / float(max(1, crop_w))
        perspective_ratio = width_top_norm / max(width_bottom_norm, 1e-6)
        width_score = 1.0
        if width_bottom_norm < 0.20 or width_bottom_norm > 0.98:
            width_score *= 0.65
        if perspective_ratio < 0.08 or perspective_ratio > 0.95:
            width_score *= 0.65
        if top_width <= 2 or bottom_width <= 4:
            width_score *= 0.45

        geometry_score = 0.40 * coverage_score + 0.30 * parallel_score + 0.30 * width_score
        confidence = detector_conf * (0.40 + 0.60 * geometry_score)
    elif left_best is not None or right_best is not None:
        one_side_cov = float((left_best or right_best)["coverage"])  # type: ignore[index]
        confidence = detector_conf * 0.35 * (0.50 + 0.50 * one_side_cov)
    else:
        confidence = 0.0

    geometry["geometry_confidence"] = round(min(1.0, max(0.0, confidence)), 4)
    return geometry


if __name__ == "__main__":
    from pathlib import Path

    from runway_detector import detect_runway

    base_dir = Path(__file__).resolve().parents[3]
    test_img_path = base_dir / "backend" / "experiments" / "test2.jpeg"
    out_path = base_dir / "backend" / "experiments" / "geometry_test_output.png"

    img = cv2.imread(str(test_img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {test_img_path}")

    detection = detect_runway(img)
    geometry = estimate_runway_geometry(img, detection)
    print("Geometry:", geometry)

    out = draw_runway_overlay(img, geometry, frame_index=0, detector_confidence=detection.get("confidence"))
    cv2.imwrite(str(out_path), out)
    print(f"Saved visualization to: {out_path}")
