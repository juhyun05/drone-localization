"""Compare fixed and automatic span matching on identical height-sweep frames."""

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from localization.camera_interface import VideoFileSource
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.map_loader import BACKGROUND, CampusMap, MapRaster
from localization.map_matching import Estimate, MapMatcher, SearchConfig, angle_error
from localization.scale_cli import add_scale_arguments, scale_arguments
from localization.tracking_manager import TrackingManager
from tools.calibrate_camera import read_image, write_image
from tools.inspect_camera_frame import capture_frame


@dataclass(frozen=True)
class HeightSample:
    scenario: str
    frame: np.ndarray
    height_m: float | None
    ratio: float | None
    truth_xy: tuple[float, float] | None
    truth_yaw: float | None
    truth_span_m: float | None


def synthetic_samples(campus):
    camera = CameraConfig()
    raster = MapRaster(campus, 320)
    pose = Pose(1.73, 2.02, 37)

    def sample(scenario, ratio, at=pose, blur=0, **options):
        config = replace(camera, height_m=camera.height_m * ratio, **options)
        frame = SyntheticCamera(raster, config).capture(at)
        if blur:
            frame = cv2.GaussianBlur(frame, (blur, blur), 1.8)
        return HeightSample(scenario, frame, config.height_m, ratio, (at.x_m, at.y_m),
                            at.yaw_deg, config.footprint_m)

    comparison = [sample("height sweep", ratio)
                  for ratio in (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75)]
    comparison.extend([
        sample("near map edge", 1.30, Pose(2.40, 1.70, 25)),
        sample("few buildings", 0.65, Pose(0.55, 1.90, 18)),
        sample("rotated, noise, blur and mild color shift", 1.35, Pose(1.73, 2.02, 112),
               blur=7, noise_std=2, brightness=0.93, channel_gains=(1.01, 0.995, 1.01)),
        sample("strong color shift", 1.35, Pose(1.73, 2.02, 112),
               blur=7, noise_std=3, brightness=0.90, channel_gains=(1.03, 0.96, 1.04)),
    ])
    level = sample("strong perspective tilt", 1.25)
    source = np.float32([[0, 0], [239, 0], [239, 179], [0, 179]])
    tilted = np.float32([[92, 6], [147, 6], [239, 179], [0, 179]])
    transform = cv2.getPerspectiveTransform(source, tilted)
    frame = cv2.warpPerspective(level.frame, transform, (240, 180), borderValue=BACKGROUND)
    comparison.append(replace(level, frame=frame, truth_span_m=None))

    gradual = [sample("gradual", ratio) for ratio in (1, 1.06, 1.12, 1.20, 1.28, 1.36, 1.30)]
    jump = sample("height jump", 1.75)
    # One abrupt jump with the same pose, then a blank gap that forces a true global reset.
    blank = replace(jump, scenario="occluded", frame=np.full_like(jump.frame, 255))
    tracking = gradual + [jump] * 4 + [blank] * 3 + [sample("after occlusion", 0.60)] * 3
    return comparison, tracking


def estimate_row(sample, estimate, elapsed_ms, mode, image_name, physical_per_map):
    pos_error = yaw_error = span_error = None
    if estimate.success:
        if sample.truth_xy is not None:
            pos_error = 100 * math.dist((estimate.x_m, estimate.y_m), sample.truth_xy)
        if sample.truth_yaw is not None:
            yaw_error = angle_error(estimate.yaw_deg, sample.truth_yaw)
        if sample.truth_span_m is not None:
            span_error = 100 * abs(estimate.span_m - sample.truth_span_m)
    estimate_physical = (estimate.span_m * physical_per_map
                         if estimate.span_m is not None and physical_per_map is not None else None)
    reference_physical = (sample.truth_span_m * physical_per_map
                          if sample.truth_span_m is not None and physical_per_map is not None else None)
    return dict(
        scenario=sample.scenario, camera_frame=str(image_name), mode=mode, height_m=sample.height_m,
        height_ratio=sample.ratio, true_x_m=sample.truth_xy[0] if sample.truth_xy else None,
        true_y_m=sample.truth_xy[1] if sample.truth_xy else None, true_yaw_deg=sample.truth_yaw,
        true_span_map_m=sample.truth_span_m, true_span_physical_m=reference_physical,
        success=estimate.success, reason=estimate.reason, x_m=estimate.x_m, y_m=estimate.y_m,
        yaw_deg=estimate.yaw_deg, span_map_m=estimate.span_m, span_physical_m=estimate_physical,
        iou=estimate.score, margin=estimate.margin, position_error_cm=pos_error,
        yaw_error_deg=yaw_error, span_error_cm=span_error, processing_ms=elapsed_ms,
    )


def test_height_robustness(samples, matchers, output, *, preprocessor=None, physical_per_map=None):
    """Evaluate fixed and automatic matching without giving either the labeled truths."""
    rows = []
    for i, sample in enumerate(samples, start=1):
        name = Path(f"frames/{i:03d}.png")
        write_image(output / name, sample.frame)
        image = preprocessor.apply(sample.frame) if preprocessor else sample.frame
        for mode, matcher in matchers.items():
            start = perf_counter()
            estimate = matcher.estimate(image)
            ms = (perf_counter() - start) * 1000
            rows.append(estimate_row(sample, estimate, ms, mode, name, physical_per_map))
    return rows


def tracking_sweep(samples, matchers, output):
    trackers = {mode: TrackingManager(matcher) for mode, matcher in matchers.items()}
    rows = []
    for i, sample in enumerate(samples):
        name = Path(f"tracking_frames/{i + 1:03d}.png")
        write_image(output / name, sample.frame)
        for mode, tracker in trackers.items():
            result = tracker.step(sample.frame, i * 0.25)
            estimate = result.measurement
            if estimate is None or not result.measurement_used:
                estimate = Estimate(False, result.reason, estimate.score if estimate else 0,
                                    estimate.margin if estimate else 0)
            row = estimate_row(sample, estimate, result.match_ms, mode, name, None)
            row.update(timestamp_s=result.timestamp_s, state=result.state.value, attempt=result.attempt,
                       relocalized=result.relocalized, tracked_span_map_m=result.span_m,
                       track_error_cm=(100 * math.dist(result.xy_m, sample.truth_xy)
                                       if result.xy_m is not None else None))
            rows.append(row)
    return rows


def publish(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    count = len(rows)
    accepted = [row for row in rows if row["success"]]
    position = [row["position_error_cm"] for row in accepted if row["position_error_cm"] is not None]
    yaw = [row["yaw_error_deg"] for row in accepted if row["yaw_error_deg"] is not None]
    span = [row["span_error_cm"] for row in accepted if row["span_error_cm"] is not None]
    times = [row["processing_ms"] for row in rows]
    return dict(
        accepted=len(accepted), count=count, success_rate=len(accepted) / count if count else None,
        position_rmse_cm=math.sqrt(mean(value ** 2 for value in position)) if position else None,
        mean_yaw_error_deg=mean(yaw) if yaw else None, mean_span_error_cm=mean(span) if span else None,
        mean_iou=mean([row["iou"] for row in rows]) if rows else None,
        mean_ms=mean(times) if times else None, p95_ms=float(np.percentile(times, 95)) if times else None,
    )


def video_frames(path, indices):
    source = VideoFileSource(path)
    try:
        last = 0
        for index in indices:
            for _ in range(last, index):
                frame = source.read()
                if frame is None:
                    raise ValueError(f"{path} ended before frame {index}")
            yield frame.image
            last = index
    finally:
        source.close()


def physical_samples(args, physical_per_map):
    if args.images:
        count = len(args.images)
    elif args.video_file:
        indices = args.frame_indices or [1]
        count = len(indices)
    else:
        count = len(args.heights_m) if args.heights_m else args.capture_count
    heights = args.heights_m or [None] * count
    widths = args.observed_widths_m or [None] * count
    if len(heights) != count or len(widths) != count:
        raise ValueError("supply one height and observed width per frame when labeling them")
    if args.observed_widths_m and physical_per_map is None:
        raise ValueError("--observed-widths-m requires --print-width-m")
    def collect():
        for i, height in enumerate(heights):
            label = f" at {height:g} m" if height is not None else ""
            input(f"Set the camera{label} without shifting x, y or yaw. Press Enter for frame {i + 1}/{count}. ")
            yield capture_frame(args.camera_url)

    if args.images:
        frames = map(read_image, args.images)
    elif args.video_file:
        frames = video_frames(args.video_file, indices)
    else:
        frames = collect()
    for frame, height, width in zip(frames, heights, widths, strict=True):
        span = width / physical_per_map if width is not None else None
        yield HeightSample("physical", frame, height, None, args.ground_truth, args.truth_yaw, span)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images", type=Path, nargs="+")
    source.add_argument("--video-file", type=Path)
    source.add_argument("--camera-url", help="XIAO MJPEG stream; captures one frame per prompt")
    parser.add_argument("--frame-indices", type=int, nargs="+",
                        help="ordered 1-based frames from --video-file, default: first frame")
    parser.add_argument("--output", type=Path, required=True, help="new directory for frames and CSVs")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--nominal-span", type=float, help="shared reference width in map meters")
    parser.add_argument("--scales", type=float, nargs="+", default=SearchConfig().scale_factors,
                        help="fixed search multipliers")
    add_scale_arguments(parser, auto_scale=True)
    parser.add_argument("--ground-truth", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--truth-yaw", type=float)
    parser.add_argument("--heights-m", type=float, nargs="+")
    parser.add_argument("--observed-widths-m", type=float, nargs="+", help="measured physical widths")
    parser.add_argument("--print-width-m", type=float, help="actual printed width without margins")
    parser.add_argument("--capture-count", type=int, default=3)
    args = parser.parse_args()
    try:
        measurements = [*(args.heights_m or ()), *(args.observed_widths_m or ())]
        if args.print_width_m is not None:
            measurements.append(args.print_width_m)
        if args.capture_count < 1 or not all(math.isfinite(value) and value > 0 for value in measurements):
            raise ValueError("physical dimensions and capture count must be positive")
        if args.frame_indices and (not args.video_file or args.frame_indices[0] < 1
                                   or args.frame_indices != sorted(set(args.frame_indices))):
            raise ValueError("--frame-indices requires --video-file and strictly increasing positive indices")
        campus = CampusMap.load()
        if args.ground_truth is not None:
            x, y = args.ground_truth
            if not (0 <= x <= campus.width_m and 0 <= y <= campus.height_m):
                raise ValueError("--ground-truth must be inside the model map")
        if args.truth_yaw is not None and not math.isfinite(args.truth_yaw):
            raise ValueError("--truth-yaw must be finite")
        config = SearchConfig()
        ratio = args.print_width_m / campus.width_m if args.print_width_m is not None else None
        preprocessor = None
        reference = args.nominal_span
        if args.images or args.video_file or args.camera_url:
            physical = (PhysicalCameraConfig.load(args.camera_config) if args.camera_config
                        else PhysicalCameraConfig.load())
            preprocessor = CameraPreprocessor(physical)
            if reference is None:
                reference = physical.nominal_span_m
            samples, tracking = list(physical_samples(args, ratio)), []
        else:
            samples, tracking = synthetic_samples(campus)
            ratio = None
        if reference is None:
            reference = config.nominal_span_m
        fixed = replace(config, nominal_span_m=reference, scale_factors=tuple(args.scales))
        auto = replace(fixed, **scale_arguments(args))
        matchers = {"fixed": MapMatcher(campus, fixed)}
        if args.auto_scale:
            matchers["auto"] = MapMatcher(campus, auto)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "frames").mkdir()
        rows = test_height_robustness(samples, matchers, args.output,
                                      preprocessor=preprocessor, physical_per_map=ratio)
        publish(args.output / "comparison.csv", rows)
        summary = {"global": {mode: summarize([r for r in rows if r["mode"] == mode])
                              for mode in matchers}}
        if tracking:
            (args.output / "tracking_frames").mkdir()
            trace = tracking_sweep(tracking, matchers, args.output)
            publish(args.output / "tracking.csv", trace)
            summary["tracking"] = {}
            for mode in matchers:
                mode_trace = [row for row in trace if row["mode"] == mode]
                summary["tracking"][mode] = {
                    "all_frames": summarize(mode_trace),
                    **{attempt: summarize([row for row in mode_trace if row["attempt"] == attempt])
                       for attempt in ("local", "global", "none")},
                }
        serialized = json.dumps(summary, indent=2)
        (args.output / "summary.json").write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    except (EOFError, OSError, ValueError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
