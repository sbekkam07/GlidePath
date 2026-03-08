from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _line_x_at_y(line: Tuple[int, int, int, int], y: int) -> int:
    x1, y1, x2, y2 = line
    if y2 == y1:
        return x1
    t = (y - y1) / float(y2 - y1)
    x = x1 + t * (x2 - x1)
    return int(round(x))


def _clip_line_to_crop(
    line: Tuple[int, int, int, int],
    crop_h: int,
) -> Tuple[int, int, int, int]:
    y_top = 0
    y_bottom = crop_h - 1
    x_top = _line_x_at_y(line, y_top)
    x_bottom = _line_x_at_y(line, y_bottom)
    return (x_top, y_top, x_bottom, y_bottom)


def estimate_runway_geometry(
    image: np.ndarray,
    detection: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Input:
        image: full OpenCV image
        detection: {"bbox": [x1,y1,x2,y2], "confidence": float}

    Output:
        {
            "left_edge": [(x1,y1), (x2,y2)] or None,
            "right_edge": [(x1,y1), (x2,y2)] or None,
            "centerline": [(x1,y1), (x2,y2)] or None,
            "glidepath_line": [(x1,y1), (x2,y2)] or None,
            "runway_center_bottom_x": int or None,
            "image_center_x": int,
            "offset_px": int or None,
            "annotated_image": np.ndarray,
        }
    """
    annotated = image.copy()
    h, w = image.shape[:2]
    image_center_x = w // 2

    bbox = detection.get("bbox")
    if bbox is None:
        return {
            "left_edge": None,
            "right_edge": None,
            "centerline": None,
            "glidepath_line": None,
            "runway_center_bottom_x": None,
            "image_center_x": image_center_x,
            "offset_px": None,
            "annotated_image": annotated,
        }

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return {
            "left_edge": None,
            "right_edge": None,
            "centerline": None,
            "glidepath_line": None,
            "runway_center_bottom_x": None,
            "image_center_x": image_center_x,
            "offset_px": None,
            "annotated_image": annotated,
        }

    crop_h, crop_w = crop.shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=max(30, crop_h // 5),
        maxLineGap=30,
    )

    left_candidates = []
    right_candidates = []

    if lines is not None:
        for raw in lines:
            lx1, ly1, lx2, ly2 = raw[0]

            if lx2 == lx1:
                continue

            slope = (ly2 - ly1) / float(lx2 - lx1)
            length = np.hypot(lx2 - lx1, ly2 - ly1)

            if abs(slope) < 0.5:
                continue

            mid_x = (lx1 + lx2) / 2.0

            if slope < 0 and mid_x < crop_w * 0.55:
                left_candidates.append((length, (lx1, ly1, lx2, ly2)))
            elif slope > 0 and mid_x > crop_w * 0.45:
                right_candidates.append((length, (lx1, ly1, lx2, ly2)))

    left_line = max(left_candidates, key=lambda x: x[0])[1] if left_candidates else None
    right_line = max(right_candidates, key=lambda x: x[0])[1] if right_candidates else None

    left_edge = None
    right_edge = None
    centerline = None
    glidepath_line = None
    runway_center_bottom_x = None
    offset_px = None

    # draw detection bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if left_line is not None:
        lxt, lyt, lxb, lyb = _clip_line_to_crop(left_line, crop_h)
        left_edge = [(x1 + lxt, y1 + lyt), (x1 + lxb, y1 + lyb)]
        cv2.line(annotated, left_edge[0], left_edge[1], (255, 0, 0), 3)

    if right_line is not None:
        rxt, ryt, rxb, ryb = _clip_line_to_crop(right_line, crop_h)
        right_edge = [(x1 + rxt, y1 + ryt), (x1 + rxb, y1 + ryb)]
        cv2.line(annotated, right_edge[0], right_edge[1], (0, 0, 255), 3)

    if left_edge is not None and right_edge is not None:
        center_top_x = (left_edge[0][0] + right_edge[0][0]) // 2
        center_top_y = (left_edge[0][1] + right_edge[0][1]) // 2

        center_bottom_x = (left_edge[1][0] + right_edge[1][0]) // 2
        center_bottom_y = (left_edge[1][1] + right_edge[1][1]) // 2

        centerline = [(center_top_x, center_top_y), (center_bottom_x, center_bottom_y)]
        cv2.line(annotated, centerline[0], centerline[1], (0, 255, 255), 3)

        # simple glidepath reference: from image bottom center to runway vanishing direction
        glidepath_line = [(image_center_x, h - 1), (center_top_x, center_top_y)]
        cv2.line(annotated, glidepath_line[0], glidepath_line[1], (255, 255, 0), 2)

        runway_center_bottom_x = center_bottom_x
        offset_px = runway_center_bottom_x - image_center_x

        cv2.line(
            annotated,
            (image_center_x, h - 1),
            (image_center_x, max(0, h - 120)),
            (200, 200, 200),
            2,
        )

        cv2.putText(
            annotated,
            f"offset_px: {offset_px}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return {
        "left_edge": left_edge,
        "right_edge": right_edge,
        "centerline": centerline,
        "glidepath_line": glidepath_line,
        "runway_center_bottom_x": runway_center_bottom_x,
        "image_center_x": image_center_x,
        "offset_px": offset_px,
        "annotated_image": annotated,
    }


if __name__ == "__main__":
    from pathlib import Path
    from runway_detector import detect_runway

    base_dir = Path(__file__).resolve().parents[3]
    test_img_path = base_dir / "backend" / "experiments" / "test1.jpeg"
    out_path = base_dir / "backend" / "experiments" / "geometry_test_output.png"

    img = cv2.imread(str(test_img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {test_img_path}")

    detection = detect_runway(img)
    print("Detection:", detection)

    geometry = estimate_runway_geometry(img, detection)
    print("Offset:", geometry["offset_px"])

    cv2.imwrite(str(out_path), geometry["annotated_image"])
    print(f"Saved visualization to: {out_path}")