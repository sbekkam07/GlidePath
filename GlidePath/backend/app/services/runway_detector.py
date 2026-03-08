"""High-level runway detection orchestration."""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict

import cv2
import numpy as np

from ..utils.video import inspect_video_file, render_overlay_frame
from .runway_detector_yolo import detect_runway_frame, load_model
from .runway_geometry import estimate_runway_geometry


LOGGER = logging.getLogger(__name__)

# Project root: GlidePath/GlidePath
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "best.pt"

SMOOTH_OFFSET_WINDOW = 5
SMOOTH_GEOMETRY_ALPHA = 0.55
SMOOTH_BBOX_ALPHA = 0.45
WEAK_GEOMETRY_REUSE_LIMIT = 3
MISSING_BBOX_REUSE_LIMIT = 3
MISSING_CONFIDENCE_DECAY = 0.88
GEOMETRY_STRENGTH_THRESHOLD = 0.58

SAFE_FPS_FALLBACK = 24.0
MP4_CODEC_PREFERENCES = ("avc1", "H264", "mp4v")
AUX_CODEC_PREFERENCES = ("MJPG", "XVID")
BROWSER_FRIENDLY_CODECS = {"avc1", "h264"}

# Lazy singleton to keep app startup resilient when model weights are missing/corrupt.
_MODEL = None
_MODEL_LOAD_ERROR: Exception | None = None


def _get_model():
    """Load YOLO model once on first use and cache it."""
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        raise _MODEL_LOAD_ERROR

    try:
        _MODEL = load_model(MODEL_PATH)
        return _MODEL
    except Exception as exc:
        _MODEL_LOAD_ERROR = RuntimeError(
            f"Unable to load YOLO model from '{MODEL_PATH}'. "
            "If this is a Git LFS pointer or placeholder, fetch the real weights file."
        )
        raise _MODEL_LOAD_ERROR from exc


def _empty_geometry(image_width: int | None = None, bbox: list[int] | None = None) -> Dict[str, Any]:
    return {
        "bbox": list(bbox) if bbox else None,
        "left_edge": None,
        "right_edge": None,
        "centerline": None,
        "center_x_bottom": None,
        "image_center_x": int(image_width // 2) if image_width else 0,
        "signed_offset_px": None,
        "runway_polygon": None,
        "geometry_confidence": 0.0,
    }


def _empty_detection(frame: int | None = None, image_width: int | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "bbox": None,
        "confidence": 0.0,
        "center_x": None,
        "center_y": None,
        "width": None,
        "height": None,
        "geometry": _empty_geometry(image_width),
    }
    if frame is not None:
        payload["frame"] = frame
    return payload


def _select_best_detection(detections: list[dict]) -> Dict[str, Any]:
    if not detections:
        return _empty_detection()
    best = max(detections, key=lambda d: float(d.get("confidence", 0.0) or 0.0))
    return {
        "bbox": best.get("bbox"),
        "confidence": float(best.get("confidence", 0.0) or 0.0),
        "center_x": best.get("center_x"),
        "center_y": best.get("center_y"),
        "width": best.get("width"),
        "height": best.get("height"),
        "geometry": best.get("geometry"),
    }


def _is_geometry_strong(geometry: dict[str, Any]) -> bool:
    return (
        isinstance(geometry, dict)
        and geometry.get("left_edge") is not None
        and geometry.get("right_edge") is not None
        and float(geometry.get("geometry_confidence", 0.0) or 0.0) >= GEOMETRY_STRENGTH_THRESHOLD
    )


def _blend_line(
    previous: list[int] | tuple[int, int, int, int] | None,
    current: list[int] | tuple[int, int, int, int] | None,
    alpha: float,
) -> list[int] | None:
    if current is None:
        return list(previous) if previous is not None else None
    if previous is None:
        return list(current)

    prev = [float(v) for v in previous]
    curr = [float(v) for v in current]
    smoothed = [int(round((1.0 - alpha) * p + alpha * c)) for p, c in zip(prev, curr)]
    return smoothed


def _blend_bbox(
    previous: list[int] | None,
    current: list[int] | None,
    alpha: float,
) -> list[int] | None:
    if current is None:
        return list(previous) if previous is not None else None
    if previous is None:
        return list(current)
    if len(previous) != 4 or len(current) != 4:
        return list(current)

    blended = [int(round((1.0 - alpha) * float(p) + alpha * float(c))) for p, c in zip(previous, current)]
    x1, y1, x2, y2 = blended
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _detection_from_bbox(bbox: list[int], confidence: float) -> Dict[str, Any]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return {
        "confidence": float(max(0.0, min(1.0, confidence))),
        "bbox": [x1, y1, x2, y2],
        "center_x": (x1 + x2) // 2,
        "center_y": (y1 + y2) // 2,
        "width": width,
        "height": height,
    }


def _recompute_centerline_and_offset(geometry: Dict[str, Any]) -> None:
    left_edge = geometry.get("left_edge")
    right_edge = geometry.get("right_edge")
    if not (isinstance(left_edge, list) and len(left_edge) == 4):
        geometry["centerline"] = None
        geometry["center_x_bottom"] = None
        geometry["signed_offset_px"] = None
        geometry["runway_polygon"] = None
        return
    if not (isinstance(right_edge, list) and len(right_edge) == 4):
        geometry["centerline"] = None
        geometry["center_x_bottom"] = None
        geometry["signed_offset_px"] = None
        geometry["runway_polygon"] = None
        return

    image_center_x = float(geometry.get("image_center_x", 0))
    left_top_x, left_top_y, left_bottom_x, left_bottom_y = left_edge
    right_top_x, right_top_y, right_bottom_x, right_bottom_y = right_edge

    center_top_x = (left_top_x + right_top_x) / 2.0
    center_top_y = (left_top_y + right_top_y) / 2.0
    center_bottom_x = (left_bottom_x + right_bottom_x) / 2.0
    center_bottom_y = (left_bottom_y + right_bottom_y) / 2.0

    geometry["centerline"] = [
        int(round(center_top_x)),
        int(round(center_top_y)),
        int(round(center_bottom_x)),
        int(round(center_bottom_y)),
    ]
    geometry["center_x_bottom"] = float(center_bottom_x)
    geometry["signed_offset_px"] = float(center_bottom_x - image_center_x)
    geometry["runway_polygon"] = [
        [int(round(left_bottom_x)), int(round(left_bottom_y))],
        [int(round(right_bottom_x)), int(round(right_bottom_y))],
        [int(round(right_top_x)), int(round(right_top_y))],
        [int(round(left_top_x)), int(round(left_top_y))],
    ]


def _stabilize_geometry(
    current: Dict[str, Any],
    previous_smoothed: Dict[str, Any] | None,
) -> Dict[str, Any]:
    geometry = dict(current)
    if previous_smoothed is None:
        _recompute_centerline_and_offset(geometry)
        return geometry

    alpha = SMOOTH_GEOMETRY_ALPHA
    geometry["left_edge"] = _blend_line(
        previous_smoothed.get("left_edge"),
        geometry.get("left_edge"),
        alpha,
    )
    geometry["right_edge"] = _blend_line(
        previous_smoothed.get("right_edge"),
        geometry.get("right_edge"),
        alpha,
    )
    _recompute_centerline_and_offset(geometry)
    prev_conf = float(previous_smoothed.get("geometry_confidence", 0.0) or 0.0)
    curr_conf = float(geometry.get("geometry_confidence", 0.0) or 0.0)
    geometry["geometry_confidence"] = round((0.35 * prev_conf + 0.65 * curr_conf), 4)
    return geometry


def _reuse_geometry_if_weak(
    geometry: Dict[str, Any],
    cached_reference: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if cached_reference is None:
        return geometry

    merged = dict(geometry)
    if merged.get("left_edge") is None:
        merged["left_edge"] = cached_reference.get("left_edge")
    else:
        merged["left_edge"] = _blend_line(cached_reference.get("left_edge"), merged.get("left_edge"), 0.5)

    if merged.get("right_edge") is None:
        merged["right_edge"] = cached_reference.get("right_edge")
    else:
        merged["right_edge"] = _blend_line(cached_reference.get("right_edge"), merged.get("right_edge"), 0.5)

    merged["geometry_confidence"] = float(merged.get("geometry_confidence", 0.0) or 0.0) * 0.60
    if merged.get("bbox") is None and cached_reference.get("bbox") is not None:
        merged["bbox"] = list(cached_reference["bbox"])
    _recompute_centerline_and_offset(merged)
    return merged


def _sanitize_fps(raw_fps: float) -> float:
    if not isinstance(raw_fps, (int, float)):
        return SAFE_FPS_FALLBACK
    fps = float(raw_fps)
    if not math.isfinite(fps) or fps <= 1.0:
        return SAFE_FPS_FALLBACK
    return fps


def _open_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    codec_preferences: tuple[str, ...],
) -> tuple[cv2.VideoWriter | None, str | None]:
    for codec in codec_preferences:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer, codec
            writer.release()
        except Exception:
            continue
    return None, None


def _transcode_with_ffmpeg(source: Path, target: Path) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        LOGGER.warning(
            "ffmpeg invocation failed for '%s' -> '%s': %s",
            source,
            target,
            exc,
        )
        return False
    if completed.returncode != 0:
        LOGGER.warning(
            "ffmpeg transcode failed for '%s' -> '%s' (code=%s, stderr=%s)",
            source,
            target,
            completed.returncode,
            completed.stderr.strip()[-400:],
        )
        return False
    return True


def _transcode_with_opencv(source: Path, target: Path, fps: float, width: int, height: int) -> bool:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return False

    writer, codec = _open_video_writer(target, fps, width, height, MP4_CODEC_PREFERENCES)
    if writer is None:
        cap.release()
        return False

    success_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame is None or frame.size == 0:
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
        success_count += 1

    cap.release()
    writer.release()
    LOGGER.info(
        "OpenCV transcode complete: '%s' -> '%s' frames=%d codec=%s",
        source,
        target,
        success_count,
        codec or "unknown",
    )
    info = inspect_video_file(target)
    return bool(info["exists"] and info["size_bytes"] > 0 and info["readable"])


def _finalize_mp4_output(
    source_video_path: Path,
    final_output_path: Path,
    fps: float,
    width: int,
    height: int,
    selected_codec: str | None,
) -> None:
    if source_video_path != final_output_path:
        ffmpeg_ok = _transcode_with_ffmpeg(source_video_path, final_output_path)
        if not ffmpeg_ok:
            opencv_ok = _transcode_with_opencv(source_video_path, final_output_path, fps, width, height)
            if not opencv_ok:
                LOGGER.error(
                    "Failed to produce final MP4 from temp file '%s' -> '%s'",
                    source_video_path,
                    final_output_path,
                )
        return

    if not selected_codec:
        return

    if selected_codec.lower() in BROWSER_FRIENDLY_CODECS:
        return

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return

    tmp_h264 = final_output_path.with_name(f"{final_output_path.stem}_h264.mp4")
    ffmpeg_ok = _transcode_with_ffmpeg(final_output_path, tmp_h264)
    if not ffmpeg_ok:
        if tmp_h264.exists():
            tmp_h264.unlink(missing_ok=True)
        return

    info = inspect_video_file(tmp_h264)
    if info["exists"] and info["size_bytes"] > 0 and info["readable"]:
        final_output_path.unlink(missing_ok=True)
        tmp_h264.replace(final_output_path)
        LOGGER.info(
            "Replaced non-browser codec MP4 with H264 output: '%s'",
            final_output_path,
        )
    else:
        tmp_h264.unlink(missing_ok=True)


def detect_runway(image_or_video: np.ndarray | str | Path) -> Dict[str, Any]:
    """Backward-compatible entrypoint used by existing routes.

    - numpy frame input: return a frame-level detection dict
    - path input (str/Path): run video inference and return best detection across frames
    """
    model = _get_model()

    if isinstance(image_or_video, (str, Path)):
        detections = detect_runway_series(str(image_or_video))
        return _select_best_detection(detections)

    width = image_or_video.shape[1] if isinstance(image_or_video, np.ndarray) and image_or_video.ndim >= 2 else 0
    frame_detection = detect_runway_frame(model, image_or_video) or _empty_detection(image_width=width)
    if frame_detection.get("bbox") is not None:
        frame_detection["frame"] = 0
        try:
            frame_detection["geometry"] = estimate_runway_geometry(image_or_video, frame_detection)
        except Exception:
            frame_detection["geometry"] = _empty_geometry(width, frame_detection.get("bbox"))
    else:
        frame_detection["geometry"] = _empty_geometry(width)
    return frame_detection


def detect_runway_series(
    video_path: str | Path,
    annotated_output_path: str | Path | None = None,
) -> list[Dict[str, Any]]:
    """Run runway detection + geometry on each frame and return per-frame detections."""
    model = _get_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.error("Could not open input video: %s", video_path)
        return []

    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = _sanitize_fps(fps_raw)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_center_x = width // 2 if width > 0 else 0

    writer: cv2.VideoWriter | None = None
    selected_codec: str | None = None
    writer_source_path: Path | None = None
    final_output_path = Path(annotated_output_path) if annotated_output_path is not None else None
    writer_enabled = False
    temp_work_dir: Path | None = None

    if final_output_path is not None and width > 0 and height > 0:
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        writer, selected_codec = _open_video_writer(
            final_output_path,
            fps,
            width,
            height,
            MP4_CODEC_PREFERENCES,
        )
        writer_source_path = final_output_path
        if writer is None:
            temp_work_dir = Path(tempfile.mkdtemp(prefix="glidepath_overlay_"))
            temp_avi_path = temp_work_dir / "overlay_temp.avi"
            writer, selected_codec = _open_video_writer(
                temp_avi_path,
                fps,
                width,
                height,
                AUX_CODEC_PREFERENCES,
            )
            writer_source_path = temp_avi_path

        writer_enabled = writer is not None and writer.isOpened()
        LOGGER.info(
            "Annotated writer init path=%s size=%dx%d fps_in=%.2f fps_used=%.2f opened=%s codec=%s",
            final_output_path,
            width,
            height,
            fps_raw,
            fps,
            writer_enabled,
            selected_codec or "none",
        )
    elif final_output_path is not None:
        LOGGER.warning(
            "Annotated writer skipped due to invalid frame size path=%s size=%dx%d",
            final_output_path,
            width,
            height,
        )

    results_out: list[Dict[str, Any]] = []
    frame_index = 0
    stable_offset_window: Deque[float] = deque(maxlen=SMOOTH_OFFSET_WINDOW)

    previous_geometry: Dict[str, Any] | None = None
    last_valid_geometry: Dict[str, Any] | None = None
    last_valid_frame = -10_000

    previous_bbox: list[int] | None = None
    previous_bbox_conf = 0.0
    missed_detection_frames = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            output = _empty_detection(frame=frame_index, image_width=width)
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                results_out.append(output)
                frame_index += 1
                if writer_enabled and writer is not None:
                    writer.write(np.zeros((height, width, 3), dtype=np.uint8))
                continue

            raw_detection = detect_runway_frame(model, frame)
            using_reused_bbox = False
            detection = None

            if raw_detection is not None and raw_detection.get("bbox") is not None:
                smoothed_bbox = _blend_bbox(previous_bbox, raw_detection.get("bbox"), SMOOTH_BBOX_ALPHA)
                detection = _detection_from_bbox(smoothed_bbox, float(raw_detection.get("confidence", 0.0) or 0.0))
                previous_bbox = list(smoothed_bbox)
                previous_bbox_conf = float(detection.get("confidence", 0.0) or 0.0)
                missed_detection_frames = 0
            else:
                missed_detection_frames += 1
                if previous_bbox is not None and missed_detection_frames <= MISSING_BBOX_REUSE_LIMIT:
                    reused_conf = previous_bbox_conf * (MISSING_CONFIDENCE_DECAY**missed_detection_frames)
                    detection = _detection_from_bbox(previous_bbox, reused_conf)
                    using_reused_bbox = True

            if detection is not None:
                output.update(detection)
                output["frame"] = frame_index

                try:
                    geometry = estimate_runway_geometry(frame, detection)
                except Exception:
                    geometry = _empty_geometry(width, detection.get("bbox"))

                if (
                    not _is_geometry_strong(geometry)
                    and last_valid_geometry is not None
                    and (frame_index - last_valid_frame) <= WEAK_GEOMETRY_REUSE_LIMIT
                ):
                    geometry = _reuse_geometry_if_weak(geometry, last_valid_geometry)

                geometry = _stabilize_geometry(
                    geometry,
                    previous_geometry if previous_geometry is not None else None,
                )

                if geometry.get("bbox") is None and detection.get("bbox") is not None:
                    geometry["bbox"] = list(detection["bbox"])
                if geometry.get("image_center_x") is None:
                    geometry["image_center_x"] = image_center_x

                offset = geometry.get("signed_offset_px") if _is_geometry_strong(geometry) else None
                if not isinstance(offset, (int, float)) and detection.get("center_x") is not None:
                    offset = float(detection["center_x"] - image_center_x)
                if isinstance(offset, (int, float)):
                    stable_offset_window.append(float(offset))
                    geometry["signed_offset_px"] = float(mean(stable_offset_window))

                if _is_geometry_strong(geometry):
                    last_valid_geometry = dict(geometry)
                    last_valid_frame = frame_index
                    previous_geometry = dict(geometry)
                elif geometry.get("left_edge") is not None or geometry.get("right_edge") is not None:
                    previous_geometry = dict(geometry)
                elif using_reused_bbox and last_valid_geometry is not None:
                    previous_geometry = dict(last_valid_geometry)

                output["geometry"] = geometry
            else:
                output["geometry"] = _empty_geometry(width)
                output["frame"] = frame_index

            if writer_enabled and writer is not None:
                overlay_frame = render_overlay_frame(
                    frame,
                    frame_index=frame_index,
                    detection=output,
                    alignment=None,
                    fallback_offset=(
                        float(output.get("center_x", 0) - image_center_x)
                        if output.get("center_x") is not None
                        else None
                    ),
                )
                if overlay_frame is None:
                    overlay_frame = frame
                if overlay_frame.shape[0] != height or overlay_frame.shape[1] != width:
                    overlay_frame = cv2.resize(overlay_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                writer.write(overlay_frame)

            results_out.append(output)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

        if final_output_path is not None and writer_source_path is not None and writer_enabled:
            try:
                _finalize_mp4_output(
                    writer_source_path,
                    final_output_path,
                    fps,
                    width,
                    height,
                    selected_codec,
                )
            except Exception:
                LOGGER.exception(
                    "Annotated output finalization failed for '%s'",
                    final_output_path,
                )
            info = inspect_video_file(final_output_path)
            LOGGER.info(
                (
                    "Annotated output summary path=%s exists=%s size_bytes=%s "
                    "readable=%s frames=%s fps=%.2f width=%s height=%s"
                ),
                final_output_path,
                info["exists"],
                info["size_bytes"],
                info["readable"],
                info["frame_count"],
                float(info["fps"] or 0.0),
                info["width"],
                info["height"],
            )
        elif final_output_path is not None:
            LOGGER.warning(
                "Annotated output unavailable path=%s writer_enabled=%s",
                final_output_path,
                writer_enabled,
            )

        if temp_work_dir is not None:
            shutil.rmtree(temp_work_dir, ignore_errors=True)

    return results_out


if __name__ == "__main__":
    print("Runway detector wrapper is a runtime module; call detect_runway(...) directly.")
