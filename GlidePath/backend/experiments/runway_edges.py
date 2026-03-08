import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

Line = Tuple[int, int, int, int]


@dataclass
class FrameResult:
    left_edge: Optional[Line]
    right_edge: Optional[Line]
    centerline: Optional[Line]
    alignment: str
    offset_px: Optional[float]
    status: str
    confidence: float
    pair_score: float


@dataclass
class TrackState:
    offset_ema: Optional[float] = None
    prev_offset: Optional[float] = None
    drift_ema: float = 0.0
    conf_ema: float = 0.0


class GlidePathAnalyzer:
    def __init__(
        self,
        canny_low: int = 55,
        canny_high: int = 150,
        hough_threshold: int = 50,
        hough_min_length: int = 45,
        hough_max_gap: int = 18,
        ema_alpha: float = 0.25,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_length = hough_min_length
        self.hough_max_gap = hough_max_gap
        self.ema_alpha = ema_alpha
        self.track = TrackState()

    @staticmethod
    def _line_length(line: Line) -> float:
        x1, y1, x2, y2 = line
        return float(np.hypot(x2 - x1, y2 - y1))

    @staticmethod
    def _line_top_bottom(line: Line) -> Line:
        x1, y1, x2, y2 = line
        return (x1, y1, x2, y2) if y1 <= y2 else (x2, y2, x1, y1)

    @staticmethod
    def _x_at_y(line: Line, y: float) -> Optional[float]:
        x1, y1, x2, y2 = line
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return None
        t = (y - y1) / dy
        return x1 + t * (x2 - x1)

    @staticmethod
    def _line_intersection(l1: Line, l2: Line) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return px, py

    def _make_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Mid-band ROI for far-approach runway in your sample style.
        y_top = int(0.28 * h)
        y_bottom = int(0.78 * h)
        cx = w // 2
        top_half = int(0.10 * w)
        bottom_half = int(0.28 * w)

        poly = np.array(
            [[
                (max(0, cx - bottom_half), y_bottom),
                (max(0, cx - top_half), y_top),
                (min(w - 1, cx + top_half), y_top),
                (min(w - 1, cx + bottom_half), y_bottom),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, poly, 255)
        return mask

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        roi_mask = self._make_roi_mask((h, w))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
        return edges, roi_mask

    def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap,
        )

    def _filter_candidates(
        self, lines: Optional[np.ndarray], w: int, h: int
    ) -> Tuple[List[Line], List[Line], List[Line]]:
        left: List[Line] = []
        right: List[Line] = []
        all_good: List[Line] = []

        if lines is None:
            return left, right, all_good

        cx = w / 2
        y_ref = 0.30 * h
        y_low, y_high = 0.30 * h, 0.80 * h

        for raw in lines:
            l = self._line_top_bottom(tuple(int(v) for v in raw[0]))
            x1, y1, x2, y2 = l
            dx = x2 - x1
            dy = y2 - y1

            if self._line_length(l) < 50:
                continue
            if dy < 20:
                continue

            ang_v = np.degrees(np.arctan2(abs(dx), max(1, dy)))
            if not (8 <= ang_v <= 72):
                continue

            if not (y1 < y_high and y2 > y_low):
                continue

            x_ref = self._x_at_y(l, y_ref)
            if x_ref is None or abs(x_ref - cx) > 0.20 * w:
                continue

            all_good.append(l)
            if dx < 0 and x2 < cx:
                left.append(l)
            elif dx > 0 and x2 > cx:
                right.append(l)

        return left, right, all_good

    def _pair_score(self, l: Line, r: Line, w: int, h: int) -> float:
        inter = self._line_intersection(l, r)
        if inter is None:
            return -1e9
        vx, vy = inter

        # Vanishing point constraints.
        if not (0.0 * h <= vy <= 0.60 * h):
            return -1e9
        if not (0.15 * w <= vx <= 0.85 * w):
            return -1e9

        y_top = int(0.32 * h)
        y_bot = int(0.76 * h)
        lx_t = self._x_at_y(l, y_top)
        rx_t = self._x_at_y(r, y_top)
        lx_b = self._x_at_y(l, y_bot)
        rx_b = self._x_at_y(r, y_bot)
        if None in (lx_t, rx_t, lx_b, rx_b):
            return -1e9

        width_t = float(rx_t - lx_t)
        width_b = float(rx_b - lx_b)

        # Physical sanity: runway widens toward bottom.
        if width_t <= 0 or width_b <= 0 or width_b <= width_t:
            return -1e9
        if not (0.03 * w <= width_t <= 0.30 * w):
            return -1e9
        if not (0.08 * w <= width_b <= 0.60 * w):
            return -1e9

        center_t = (lx_t + rx_t) / 2.0
        center_b = (lx_b + rx_b) / 2.0

        # Scoring terms.
        s_vx = 1.0 - min(1.0, abs(vx - w / 2) / (0.25 * w))
        s_vy = 1.0 - min(1.0, abs(vy - 0.34 * h) / (0.25 * h))
        s_ct = 1.0 - min(1.0, abs(center_t - w / 2) / (0.20 * w))
        s_cb = 1.0 - min(1.0, abs(center_b - w / 2) / (0.40 * w))
        s_len = min(1.0, (self._line_length(l) + self._line_length(r)) / (0.9 * h))
        s_spread = min(1.0, (width_b - width_t) / (0.35 * w))

        score = (
            0.20 * s_vx
            + 0.20 * s_vy
            + 0.20 * s_ct
            + 0.10 * s_cb
            + 0.20 * s_len
            + 0.10 * s_spread
        )
        return score

    def _pick_best_pair(
        self, left: List[Line], right: List[Line], w: int, h: int
    ) -> Tuple[Optional[Line], Optional[Line], float]:
        if not left or not right:
            return None, None, -1.0

        best_s = -1e9
        best_pair: Tuple[Optional[Line], Optional[Line]] = (None, None)

        # Keep runtime sane.
        left_sorted = sorted(left, key=self._line_length, reverse=True)[:40]
        right_sorted = sorted(right, key=self._line_length, reverse=True)[:40]

        for l in left_sorted:
            for r in right_sorted:
                s = self._pair_score(l, r, w, h)
                if s > best_s:
                    best_s = s
                    best_pair = (l, r)

        if best_s < 0.20:
            return None, None, best_s
        return best_pair[0], best_pair[1], best_s

    def _extend_line_to_band(self, line: Line, h: int) -> Optional[Line]:
        y_top = int(0.28 * h)
        y_bottom = int(0.78 * h)
        x_t = self._x_at_y(line, y_top)
        x_b = self._x_at_y(line, y_bottom)
        if x_t is None or x_b is None:
            return None
        return (int(x_t), y_top, int(x_b), y_bottom)

    @staticmethod
    def _compute_centerline(left_line: Optional[Line], right_line: Optional[Line]) -> Optional[Line]:
        if left_line is None or right_line is None:
            return None
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        return (int((lx1 + rx1) / 2), min(ly1, ry1), int((lx2 + rx2) / 2), max(ly2, ry2))

    def _score_labels(
        self,
        centerline: Optional[Line],
        w: int,
        pair_score: float,
        left_ok: bool,
        right_ok: bool,
        good_lines: int,
    ) -> Tuple[str, Optional[float], str, float]:
        if centerline is None:
            self.track.conf_ema = 0.85 * self.track.conf_ema
            return "unknown", None, "unstable", self.track.conf_ema

        _, _, cx_bottom, _ = centerline
        raw_offset = float(cx_bottom - w / 2.0)

        if self.track.offset_ema is None:
            offset = raw_offset
        else:
            offset = (1 - self.ema_alpha) * self.track.offset_ema + self.ema_alpha * raw_offset

        if self.track.prev_offset is not None:
            drift = abs(offset - self.track.prev_offset)
            self.track.drift_ema = 0.8 * self.track.drift_ema + 0.2 * drift
        self.track.prev_offset = offset
        self.track.offset_ema = offset

        conf = 0.0
        conf += 0.25 if left_ok else 0.0
        conf += 0.25 if right_ok else 0.0
        conf += 0.30 * max(0.0, min(1.0, pair_score))
        conf += 0.20 * min(1.0, good_lines / 80.0)
        self.track.conf_ema = 0.75 * self.track.conf_ema + 0.25 * conf

        thr = 0.03 * w
        if abs(offset) <= thr:
            alignment = "aligned"
        elif offset > thr:
            alignment = "drifting left"
        else:
            alignment = "drifting right"

        if self.track.conf_ema < 0.45 or self.track.drift_ema > 22:
            status = "unstable"
        elif self.track.conf_ema < 0.65 or self.track.drift_ema > 10:
            status = "caution"
        else:
            status = "stable"

        return alignment, offset, status, self.track.conf_ema

    def analyze_frame(self, frame: np.ndarray) -> Tuple[FrameResult, Dict[str, Any], Dict[str, Any]]:
        h, w = frame.shape[:2]
        edges, roi_mask = self._preprocess(frame)
        raw_lines = self._detect_lines(edges)
        left_cands, right_cands, good_lines = self._filter_candidates(raw_lines, w, h)

        left_pick, right_pick, pair_score = self._pick_best_pair(left_cands, right_cands, w, h)

        left_fit = self._extend_line_to_band(left_pick, h) if left_pick is not None else None
        right_fit = self._extend_line_to_band(right_pick, h) if right_pick is not None else None
        centerline = self._compute_centerline(left_fit, right_fit)

        alignment, offset_px, status, confidence = self._score_labels(
            centerline=centerline,
            w=w,
            pair_score=pair_score,
            left_ok=left_fit is not None,
            right_ok=right_fit is not None,
            good_lines=len(good_lines),
        )

        result = FrameResult(
            left_edge=left_fit,
            right_edge=right_fit,
            centerline=centerline,
            alignment=alignment,
            offset_px=offset_px,
            status=status,
            confidence=confidence,
            pair_score=pair_score,
        )

        debug: Dict[str, Any] = {
            "edges": edges,
            "roi_mask": roi_mask,
            "raw_lines": raw_lines,
            "left_candidates": left_cands,
            "right_candidates": right_cands,
            "picked_left": left_pick,
            "picked_right": right_pick,
        }

        metrics: Dict[str, Any] = {
            "alignment": alignment,
            "status": status,
            "offset_px": offset_px,
            "confidence": confidence,
            "pair_score": pair_score,
            "left_candidates": len(left_cands),
            "right_candidates": len(right_cands),
            "good_lines": len(good_lines),
        }
        return result, debug, metrics


def draw_overlay(frame: np.ndarray, debug: Dict[str, Any], result: FrameResult, compact: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    contours, _ = cv2.findContours(debug["roi_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    raw_lines = debug["raw_lines"]
    if raw_lines is not None:
        for raw in raw_lines:
            x1, y1, x2, y2 = raw[0]
            cv2.line(out, (x1, y1), (x2, y2), (150, 150, 0), 1)

    for l in debug["left_candidates"]:
        cv2.line(out, (l[0], l[1]), (l[2], l[3]), (0, 120, 255), 2)
    for l in debug["right_candidates"]:
        cv2.line(out, (l[0], l[1]), (l[2], l[3]), (120, 255, 0), 2)

    if result.left_edge is not None:
        x1, y1, x2, y2 = result.left_edge
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 4)
    if result.right_edge is not None:
        x1, y1, x2, y2 = result.right_edge
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 4)
    if result.centerline is not None:
        x1, y1, x2, y2 = result.centerline
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 3)

    img_cx = w // 2
    cv2.line(out, (img_cx, int(0.20 * h)), (img_cx, h - 1), (255, 255, 255), 2)

    y0 = 26
    cv2.putText(out, f"alignment: {result.alignment}", (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(out, f"status: {result.status}", (12, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    cv2.putText(out, f"conf: {result.confidence:.2f}", (12, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(out, f"pair: {result.pair_score:.2f}", (12, y0 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    if result.offset_px is not None:
        cv2.putText(out, f"offset_px: {result.offset_px:.1f}", (12, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    if compact:
        return out

    edges_bgr = cv2.cvtColor(debug["edges"], cv2.COLOR_GRAY2BGR)

    def resize_keep_aspect(img: np.ndarray, new_h: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        s = new_h / ih
        return cv2.resize(img, (int(iw * s), new_h))

    target_h = 540
    a = resize_keep_aspect(frame, target_h)
    b = resize_keep_aspect(edges_bgr, target_h)
    c = resize_keep_aspect(out, target_h)
    return np.hstack([a, b, c])


def analyze_image(input_path: str, output_path: str, show: bool) -> None:
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    analyzer = GlidePathAnalyzer()
    result, debug, metrics = analyzer.analyze_frame(frame)
    panel = draw_overlay(frame, debug, result, compact=False)
    cv2.imwrite(output_path, panel)

    print(f"Saved output to: {output_path}")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if show:
        cv2.imshow("GlidePath", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def analyze_video(input_path: str, output_path: str, sample_step: int, show: bool) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    analyzer = GlidePathAnalyzer()
    i = 0
    last_result: Optional[FrameResult] = None
    last_debug: Optional[Dict[str, Any]] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if i % sample_step == 0:
            last_result, last_debug, _ = analyzer.analyze_frame(frame)

        annotated = frame if last_result is None else draw_overlay(frame, last_debug, last_result, compact=True)  # type: ignore[arg-type]
        writer.write(annotated)

        if show:
            cv2.imshow("GlidePath Video", annotated)
            if cv2.waitKey(1) == 27:
                break

        i += 1

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Saved annotated video to: {output_path}")


def is_video(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GlidePath OpenCV runway analyzer (pair-scored)")
    p.add_argument("--input", required=True, help="Image/video path")
    p.add_argument("--output", required=True, help="Output path (.png for image, .mp4 for video)")
    p.add_argument("--sample-step", type=int, default=1, help="Video: analyze every Nth frame")
    p.add_argument("--show", action="store_true", help="Show debug window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.sample_step < 1:
        raise ValueError("--sample-step must be >= 1")

    if is_video(args.input):
        if not args.output.lower().endswith(".mp4"):
            raise ValueError("Video input requires .mp4 output")
        analyze_video(args.input, args.output, args.sample_step, args.show)
    else:
        analyze_image(args.input, args.output, args.show)


if __name__ == "__main__":
    main()
