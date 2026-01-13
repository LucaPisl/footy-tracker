"""Frame extraction + stitching pipeline for Footy Tracker."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

_FRAME_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class ExtractionSummary:
    frames_dir: str
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class StitchSummary:
    output_video: str
    frame_count: int
    fps: float
    width: int
    height: int


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    config = _safe_load_yaml(config_path)
    paths_local = config.get("configs", {}).get("paths_local")
    if paths_local:
        local_path = Path(paths_local)
        if local_path.exists():
            local_config = _safe_load_yaml(local_path)
            config = _deep_merge(config, local_config)
    return config


def _require_key(config: dict[str, Any], path: str) -> Any:
    current: Any = config
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(f"Missing required config key: {path}")
        current = current[segment]
    return current


def _configure_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"
    logger = logging.getLogger("footy_tracker")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def _get_video_meta(video_path: Path) -> VideoMeta:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS detected for video: {video_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions detected: {video_path}")
    return VideoMeta(fps=fps, width=width, height=height, frame_count=frame_count)


def _compute_resize(width: int, height: int, resize_cfg: dict[str, Any]) -> tuple[int, int]:
    target_width = resize_cfg.get("width")
    target_height = resize_cfg.get("height")
    if target_width is None and target_height is None:
        return width, height
    if target_width is None:
        if target_height is None or height == 0:
            raise ValueError("Invalid resize configuration.")
        target_width = int(round(width * (target_height / height)))
    if target_height is None:
        if width == 0:
            raise ValueError("Invalid resize configuration.")
        target_height = int(round(height * (target_width / width)))
    target_width = int(target_width)
    target_height = int(target_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Resize dimensions must be positive.")
    return target_width, target_height


def _format_frame_name(pattern: str, index: int) -> str:
    if "{index" not in pattern:
        raise ValueError("frames.pattern must include an {index} placeholder.")
    try:
        return pattern.format(index=index)
    except Exception as exc:
        raise ValueError(f"Invalid frames.pattern format: {pattern}") from exc


def _frame_sort_key(path: Path) -> tuple[int, int, str]:
    match = _FRAME_INDEX_RE.search(path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def _list_frames(frames_dir: Path, output_glob: str) -> list[Path]:
    frames = sorted(frames_dir.glob(output_glob), key=_frame_sort_key)
    return [frame for frame in frames if frame.is_file()]


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    pattern: str,
    start_index: int,
    resize_cfg: dict[str, Any],
    summary_every_n_frames: int,
    logger: logging.Logger,
    meta: VideoMeta | None = None,
) -> ExtractionSummary:
    video_meta = meta or _get_video_meta(video_path)
    target_width, target_height = _compute_resize(
        video_meta.width, video_meta.height, resize_cfg
    )
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frame_index = start_index
    saved_count = 0
    log_every = summary_every_n_frames if summary_every_n_frames > 0 else None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (target_width, target_height):
            frame = cv2.resize(frame, (target_width, target_height))
        frame_name = _format_frame_name(pattern, frame_index)
        frame_path = frames_dir / frame_name
        if not cv2.imwrite(str(frame_path), frame):
            capture.release()
            raise IOError(f"Failed to write frame: {frame_path}")
        if log_every and saved_count % log_every == 0:
            logger.info("step=extract frame=%d path=%s", frame_index, frame_path)
        saved_count += 1
        frame_index += 1

    capture.release()
    if saved_count == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return ExtractionSummary(
        frames_dir=str(frames_dir),
        frame_count=saved_count,
        fps=video_meta.fps,
        width=target_width,
        height=target_height,
    )


def stitch_video(
    frames_dir: Path,
    output_video: Path,
    output_glob: str,
    fps: float,
    expected_size: tuple[int, int],
    summary_every_n_frames: int,
    logger: logging.Logger,
) -> StitchSummary:
    frames = _list_frames(frames_dir, output_glob)
    if not frames:
        raise FileNotFoundError(f"No frames found in: {frames_dir}")

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read frame: {frames[0]}")
    frame_height, frame_width = first_frame.shape[:2]
    target_width, target_height = expected_size
    if (frame_width, frame_height) != (target_width, target_height):
        logger.warning(
            "step=stitch message=frame_size_mismatch expected=%sx%s got=%sx%s",
            target_width,
            target_height,
            frame_width,
            frame_height,
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (target_width, target_height))
    if not writer.isOpened():
        raise ValueError(f"Failed to open video writer: {output_video}")

    log_every = summary_every_n_frames if summary_every_n_frames > 0 else None
    written_count = 0
    for index, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            writer.release()
            raise ValueError(f"Failed to read frame: {frame_path}")
        if (frame.shape[1], frame.shape[0]) != (target_width, target_height):
            frame = cv2.resize(frame, (target_width, target_height))
        writer.write(frame)
        if log_every and index % log_every == 0:
            logger.info("step=stitch frame=%d path=%s", index, frame_path)
        written_count += 1

    writer.release()
    return StitchSummary(
        output_video=str(output_video),
        frame_count=written_count,
        fps=fps,
        width=target_width,
        height=target_height,
    )


def run_pipeline(config_path: str) -> None:
    config = load_config(Path(config_path))

    paths_cfg = _require_key(config, "paths")
    video_cfg = _require_key(config, "video")
    frames_cfg = _require_key(config, "frames")
    steps_cfg = _require_key(config, "steps")
    logging_cfg = _require_key(config, "logging")

    input_video = Path(_require_key(paths_cfg, "input_video"))
    frames_dir = Path(_require_key(paths_cfg, "frames_dir"))
    ml_output_dir = Path(_require_key(paths_cfg, "ml_output_dir"))
    output_dir = Path(_require_key(paths_cfg, "output_dir"))
    output_name = Path(_require_key(paths_cfg, "output_name"))
    log_dir = Path(_require_key(paths_cfg, "log_dir"))

    output_video = output_name if output_name.is_absolute() else output_dir / output_name

    logger, log_path = _configure_logging(log_dir)
    logger.info("step=config path=%s log_path=%s", config_path, log_path)

    preserve_fps = bool(_require_key(video_cfg, "preserve_fps"))
    output_fps = video_cfg.get("output_fps")
    resize_cfg = _require_key(video_cfg, "resize")
    meta = _get_video_meta(input_video)
    target_width, target_height = _compute_resize(
        meta.width,
        meta.height,
        resize_cfg,
    )

    if preserve_fps:
        fps = meta.fps
    else:
        if output_fps is None:
            raise ValueError("video.output_fps must be set when preserve_fps is false.")
        fps = float(output_fps)
    if fps <= 0:
        raise ValueError("Output FPS must be positive.")

    summary_every_n_frames = int(_require_key(logging_cfg, "summary_every_n_frames"))
    extract_enabled = bool(_require_key(steps_cfg, "extract_frames"))
    stitch_enabled = bool(_require_key(steps_cfg, "stitch_video"))
    require_ml_outputs = bool(_require_key(steps_cfg, "require_ml_outputs"))

    pattern = _require_key(frames_cfg, "pattern")
    output_glob = _require_key(frames_cfg, "output_glob")
    start_index = int(_require_key(frames_cfg, "start_index"))

    extraction_summary: ExtractionSummary | None = None
    stitch_summary: StitchSummary | None = None

    if extract_enabled:
        extraction_summary = extract_frames(
            video_path=input_video,
            frames_dir=frames_dir,
            pattern=pattern,
            start_index=start_index,
            resize_cfg=resize_cfg,
            summary_every_n_frames=summary_every_n_frames,
            logger=logger,
            meta=meta,
        )
        logger.info(
            "step=extract_summary frames=%d fps=%.2f size=%sx%s",
            extraction_summary.frame_count,
            extraction_summary.fps,
            extraction_summary.width,
            extraction_summary.height,
        )

    # TODO(partner-ml): integrate ML step to populate ml_output_dir automatically.
    if require_ml_outputs and not _list_frames(ml_output_dir, output_glob):
        raise FileNotFoundError(
            "ML output frames not found. "
            f"Run the tracking/overlay step to generate PNGs in: {ml_output_dir}"
        )

    if stitch_enabled:
        stitch_summary = stitch_video(
            frames_dir=ml_output_dir,
            output_video=output_video,
            output_glob=output_glob,
            fps=fps,
            expected_size=(target_width, target_height),
            summary_every_n_frames=summary_every_n_frames,
            logger=logger,
        )
        logger.info(
            "step=stitch_summary frames=%d fps=%.2f size=%sx%s output=%s",
            stitch_summary.frame_count,
            stitch_summary.fps,
            stitch_summary.width,
            stitch_summary.height,
            stitch_summary.output_video,
        )

    summary_payload = {
        "input_video": str(input_video),
        "frames_dir": str(frames_dir),
        "ml_output_dir": str(ml_output_dir),
        "output_video": str(output_video),
        "extract": asdict(extraction_summary) if extraction_summary else None,
        "stitch": asdict(stitch_summary) if stitch_summary else None,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = log_dir / f"run_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    logger.info("step=summary path=%s", summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Footy Tracker pipeline (extract frames, then stitch ML outputs)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the pipeline YAML config (e.g. configs/pipeline.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
