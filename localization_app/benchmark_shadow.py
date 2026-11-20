"""Compare shadow and illumination methods on identical synthetic or camera frames."""

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from benchmark_scale import publish, video_frames
from localization.camera_interface import CameraFrame
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.illumination_correction import IlluminationConfig
from localization.lighting_pipeline import LightingProcessor
from localization.live_localization import LiveLocalization
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import MapMatcher, SearchConfig, angle_error
from localization.scale_cli import add_scale_arguments, scale_arguments
from simulation.lighting import LIGHTING_CASES, simulate_lighting
from tools.calibrate_camera import read_image, write_image
from tools.inspect_camera_frame import capture_frame


METHODS = {
    "baseline": ("none", False), "clahe": ("clahe", False), "constancy": ("constancy", False),
    "shadow-aware": ("none", True), "combined": ("combined", True),
}
POSES = (Pose(1.73, 2.02, 37), Pose(0.47, 1.16, 310), Pose(1.05, 0.81, 115),
         Pose(2.09, 1.71, 193), Pose(1.82, 2.26, 275))


@dataclass(frozen=True)
class LightingSample:
    case: str
    view: str
    frame: np.ndarray
    pose: Pose | None
    height_m: float | None = None
    clear: np.ndarray | None = None
    labels: np.ndarray | None = None
    truth_shadow: np.ndarray | None = None


def synthetic_samples(campus):
    raster = MapRaster(campus, 320)
    geometry = raster.labels()
    reference = CameraConfig(width_px=320, height_px=240)
    base = SyntheticCamera(raster, reference)
    taller = SyntheticCamera(raster, replace(reference, height_m=0.75))
    samples = []
    for i, (pose, camera) in enumerate([(pose, base) for pose in POSES] + [(POSES[0], taller)]):
        clear = camera.capture(pose)
        height, width = clear.shape[:2]
        labels = cv2.warpAffine(geometry, camera.map_transform(pose), (width, height),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        cases = LIGHTING_CASES if camera is base else ("clear", "soft", "complex", "severe")
        for case in cases:
            frame, shadow = simulate_lighting(clear, case)
            samples.append(LightingSample(case, f"view-{i + 1:02d}", frame, pose,
                                          camera.config.height_m, clear, labels, shadow))
    return samples


def reference_scores(matcher, observation, estimate, labels):
    if not estimate.success:
        return None, None
    aligned = matcher.project_labels(labels.shape, estimate)
    common = (labels == aligned) & (labels > 0)
    observed, predicted = labels > 0, aligned > 0
    plain = int(common.sum()) / max(1, int(observed.sum()) + int(predicted.sum()) - int(common.sum()))
    if observation.confidence is None:
        return plain, None
    confidence = observation.confidence
    intersection = confidence[common].sum()
    union = confidence[observed].sum() + confidence[predicted].sum() - intersection
    return plain, float(intersection / union) if union > 0 else 0.0


def evaluate(samples, matcher, processors, output):
    rows = []
    confusion = Counter()
    for i, sample in enumerate(samples, start=1):
        name = f"frames/{i:03d}.png"
        write_image(output / name, sample.frame)
        print(f"{i:02d}/{len(samples)} {sample.view} {sample.case}", flush=True)
        targets = sample.labels
        truth = targets > 0 if targets is not None else None
        truth_ready = truth is not None and np.any(truth)
        for mode, processor in processors.items():
            start = perf_counter()
            observation = processor.apply(sample.frame)
            prepared = perf_counter()
            estimate = matcher.estimate(observation.image, confidence_mask=observation.confidence)
            elapsed = perf_counter()
            labels = matcher.detector.detect(observation.image, include_contours=False).labels
            binary = labels != 0
            color_accuracy = binary_iou = None
            hsv_delta = (None, None, None)
            if truth_ready:
                color_accuracy = float((labels[truth] == targets[truth]).mean())
                binary_iou = float((binary & truth).sum() / (binary | truth).sum())
                pairs = np.stack((targets[truth], labels[truth]), axis=-1)
                codes, counts = np.unique(pairs, axis=0, return_counts=True)
                for (source, detected), count in zip(codes, counts):
                    confusion[sample.case, mode, int(source), int(detected)] += int(count)
                if sample.clear is not None:
                    original_hsv = cv2.cvtColor(sample.clear, cv2.COLOR_BGR2HSV)[truth].astype(np.int16)
                    output_hsv = cv2.cvtColor(observation.image, cv2.COLOR_BGR2HSV)[truth].astype(np.int16)
                    delta = np.abs(original_hsv - output_hsv)
                    delta[:, 0] = np.minimum(delta[:, 0], 180 - delta[:, 0])
                    hsv_delta = tuple(map(float, delta.mean(axis=0)))
            error = yaw_error = wrong = None
            if estimate.success and sample.pose is not None:
                error = 100 * math.dist((estimate.x_m, estimate.y_m), (sample.pose.x_m, sample.pose.y_m))
                yaw_error = angle_error(estimate.yaw_deg, sample.pose.yaw_deg) if sample.pose.yaw_deg is not None else None
                wrong = error > 5 or (yaw_error is not None and yaw_error > 15)
            score, weighted = reference_scores(matcher, observation, estimate, labels)
            shadow = observation.shadow
            shadow_iou = None
            if shadow is not None and sample.truth_shadow is not None:
                shadow_union = (shadow.shadow_mask | sample.truth_shadow).sum()
                if shadow_union:
                    shadow_iou = float((shadow.shadow_mask & sample.truth_shadow).sum() / shadow_union)
            rows.append(dict(
                view=sample.view, case=sample.case, image=name, mode=mode, height_m=sample.height_m,
                true_x_m=sample.pose.x_m if sample.pose else None,
                true_y_m=sample.pose.y_m if sample.pose else None,
                true_yaw_deg=sample.pose.yaw_deg if sample.pose else None,
                success=estimate.success, reason=estimate.reason, x_m=estimate.x_m, y_m=estimate.y_m,
                yaw_deg=estimate.yaw_deg, span_m=estimate.span_m, match_score=estimate.score,
                margin=estimate.margin, iou_at_pose=score, weighted_iou_at_pose=weighted,
                error_cm=error, yaw_error_deg=yaw_error, wrong_accepted=wrong,
                building_accuracy=color_accuracy, mask_iou=binary_iou, detected_fraction=float(binary.mean()),
                reference_fraction=float(truth.mean()) if truth is not None else None,
                mean_abs_h=hsv_delta[0], mean_abs_s=hsv_delta[1], mean_abs_v=hsv_delta[2],
                effective_coverage=shadow.effective_coverage if observation.confidence is not None else 1.0,
                shadow_ratio=shadow.shadow_ratio if shadow is not None else None,
                shadow_iou=shadow_iou, preprocess_ms=1000 * (prepared - start),
                matching_ms=1000 * (elapsed - prepared), total_ms=1000 * (elapsed - start),
            ))
    return rows, confusion


def summarize(rows):
    accepted = [row for row in rows if row["success"]]
    errors = [row["error_cm"] for row in accepted if row["error_cm"] is not None]
    wrong = sum(row["wrong_accepted"] is True for row in accepted)
    has_truth = any(row["true_x_m"] is not None for row in rows)

    def average(field, values=rows):
        known = [row[field] for row in values if row[field] is not None]
        return mean(known) if known else None

    return dict(count=len(rows), accepted=len(accepted), success_rate=len(accepted) / len(rows),
                rmse_cm=math.sqrt(mean(v * v for v in errors)) if errors else None,
                mean_yaw_error_deg=average("yaw_error_deg", accepted), wrong_accepted=wrong if has_truth else None,
                wrong_of_accepted=wrong / len(errors) if errors else None,
                wrong_of_all=wrong / len(rows) if has_truth else None,
                mean_building_accuracy=average("building_accuracy"), mean_mask_iou=average("mask_iou"),
                mean_detected_fraction=average("detected_fraction"),
                mean_effective_coverage=average("effective_coverage"),
                mean_shadow_iou=average("shadow_iou"),
                mean_match_score=average("match_score"), mean_iou_at_pose=average("iou_at_pose"),
                mean_weighted_iou_at_pose=average("weighted_iou_at_pose"),
                mean_preprocess_ms=average("preprocess_ms"), mean_matching_ms=average("matching_ms"),
                mean_total_ms=average("total_ms"))


def tracking_benchmark(campus, search, output):
    raster = MapRaster(campus, 320)
    config = CameraConfig(width_px=320, height_px=240)
    samples = []
    for i in range(18):
        pose = Pose(1.73 + i * 0.008, 2.02 - i * 0.002, 37)
        camera = SyntheticCamera(raster, replace(config, height_m=0.50 + min(i, 9) * 0.009))
        clear = camera.capture(pose)
        if i in (10, 11, 12):
            frame = np.rint(clear * 0.045).astype(np.uint8)
            case = "almost black"
        else:
            case = "clear" if i in (0, 13, 14) else "moving"
            frame, _ = simulate_lighting(clear, case, phase=(i % 8) / 7)
        samples.append((frame, pose, case, camera.config.height_m))
    directory = output / "tracking_frames"
    directory.mkdir()
    for i, (frame, _, _, _) in enumerate(samples):
        write_image(directory / f"{i + 1:03d}.png", frame)
    rows, summary = [], {}
    for name in ("baseline", "combined"):
        method, weighted = METHODS[name]
        processor = LightingProcessor(IlluminationConfig(method), shadow_aware=weighted)
        ideal = PhysicalCameraConfig(size=(320, 240), rotation=0, flip_horizontal=False, flip_vertical=False,
                                     calibration_path=None, brightness=1, gains=(1, 1, 1), nominal_span_m=None)
        localizer = LiveLocalization(MapMatcher(campus, search), CameraPreprocessor(ideal), lighting=processor)
        lost = None
        recovered = []
        mode_rows = []
        for i, (frame, pose, case, height) in enumerate(samples):
            t = i * 0.25
            result = localizer.process(CameraFrame(frame, t, i + 1, "synthetic"))
            tracked = result.tracking
            if tracked.measurement_used and lost is not None:
                recovered.append(t - lost)
                lost = None
            elif not tracked.measurement_used and lost is None and tracked.xy_m is not None:
                lost = t
            error = 100 * math.dist(tracked.xy_m, (pose.x_m, pose.y_m)) if tracked.xy_m is not None else None
            mode_rows.append(dict(mode=name, time_s=t, case=case, image=f"tracking_frames/{i + 1:03d}.png",
                                  height_m=height, state=tracked.state.value, attempt=tracked.attempt,
                                  measurement_used=tracked.measurement_used, relocalized=tracked.relocalized,
                                  error_cm=error, preprocess_ms=result.lighting.preprocess_ms,
                                  match_ms=tracked.match_ms, total_ms=result.processing_ms, reason=tracked.reason))
        rows.extend(mode_rows)
        local = [row for row in mode_rows if row["attempt"] == "local"]
        normal = [row for row in local if row["case"] == "moving" and row["measurement_used"]]
        summary[name] = dict(count=len(mode_rows), accepted=sum(row["measurement_used"] for row in mode_rows),
                             local_attempts=len(local),
                             local_total_mean_ms=mean(row["total_ms"] for row in local) if local else None,
                             matched_moving_local_mean_ms=mean(row["total_ms"] for row in normal) if normal else None,
                             recovery_s=recovered, recovery_unfinished=lost is not None)
    return rows, summary


def physical_samples(args, matcher, preprocessor, source_name):
    if args.images:
        frames = map(read_image, args.images)
        count = len(args.images)
    elif args.video_file:
        indices = args.frame_indices or [1]
        frames = video_frames(args.video_file, indices)
        count = len(indices)
    else:
        cases = args.cases or LIGHTING_CASES
        count = len(cases)

        def capture():
            for case in cases:
                input(f"Set up '{case}' without changing position or yaw, then press Enter. ")
                yield capture_frame(args.camera_url)

        frames = capture()
    cases = args.cases or (LIGHTING_CASES if args.camera_url else ["unlabeled"] * count)
    if len(cases) != count:
        raise ValueError("supply one --cases label per input frame")
    clear = preprocessor.apply(read_image(args.reference_image)) if args.reference_image else None
    labels = matcher.detector.detect(clear, include_contours=False).labels if clear is not None else None
    pose = Pose(*args.ground_truth, args.truth_yaw) if args.ground_truth else None
    samples = []
    for raw, case in zip(frames, cases, strict=True):
        frame = preprocessor.apply(raw)
        if clear is not None and frame.shape != clear.shape:
            raise ValueError("reference and sample frames must have the same preprocessed size")
        samples.append(LightingSample(case, source_name, frame, pose, clear=clear, labels=labels))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-url")
    source.add_argument("--images", type=Path, nargs="+")
    source.add_argument("--video-file", type=Path)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--cases", choices=LIGHTING_CASES, nargs="+")
    parser.add_argument("--reference-image", type=Path, help="clean frame at exactly the same pose, for evaluation")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--ground-truth", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--truth-yaw", type=float)
    parser.add_argument("--print-width-m", type=float)
    parser.add_argument("--nominal-span", type=float)
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    add_scale_arguments(parser, auto_scale=True)
    args = parser.parse_args()
    try:
        if args.print_width_m is not None and (not math.isfinite(args.print_width_m) or args.print_width_m <= 0):
            raise ValueError("--print-width-m must be positive and finite")
        if args.frame_indices and (not args.video_file or args.frame_indices[0] < 1
                                   or args.frame_indices != sorted(set(args.frame_indices))):
            raise ValueError("--frame-indices requires ordered positive indices and --video-file")
        campus = CampusMap.load()
        if args.ground_truth and not (0 <= args.ground_truth[0] <= campus.width_m
                                      and 0 <= args.ground_truth[1] <= campus.height_m):
            raise ValueError("ground truth coordinates must be inside the model map")
        if args.truth_yaw is not None and (not args.ground_truth or not math.isfinite(args.truth_yaw)):
            raise ValueError("--truth-yaw requires finite yaw and --ground-truth")
        source_name = "synthetic"
        if args.camera_url:
            source_name = "camera-stream"
        elif args.images:
            source_name = "image-files"
        elif args.video_file:
            source_name = "video-file"
        physical = source_name != "synthetic"
        search = SearchConfig(**scale_arguments(args))
        reference = args.nominal_span
        if physical:
            camera = (PhysicalCameraConfig.load(args.camera_config) if args.camera_config
                      else PhysicalCameraConfig.load())
            if reference is None:
                reference = camera.nominal_span_m
        if reference is not None:
            search = replace(search, nominal_span_m=reference)
        matcher = MapMatcher(campus, search)
        if args.output.exists():
            raise FileExistsError(f"result directory already exists: {args.output}")
        samples = (physical_samples(args, matcher, CameraPreprocessor(camera), source_name) if physical
                   else synthetic_samples(campus))
        processors = {name: LightingProcessor(IlluminationConfig(method), shadow_aware=weighted)
                      for name, (method, weighted) in METHODS.items()}
        args.output.mkdir(parents=True)
        (args.output / "frames").mkdir()
        rows, confusion = evaluate(samples, matcher, processors, args.output)
        publish(args.output / "comparison.csv", rows)
        if confusion:
            palette = ((0, 0, 0), *campus.palette)
            hex_colors = ["#{:02x}{:02x}{:02x}".format(*bgr[::-1]) for bgr in palette]
            publish(args.output / "color_confusion.csv", [
                dict(case=case, mode=mode, expected_color=hex_colors[src],
                     detected_color=hex_colors[dst] if dst else "", pixels=count)
                for (case, mode, src, dst), count in sorted(confusion.items())
            ])
        summary = dict(input=source_name, print_width_m=args.print_width_m,
                       auto_scale=args.auto_scale,
                       groups={label: {name: summarize(selected) for name in METHODS
                                       if (selected := [r for r in rows if r["mode"] == name and predicate(r)])}
                               for label, predicate in (("all", lambda r: True),
                                                        ("clear", lambda r: r["case"] == "clear"),
                                                        ("perturbed", lambda r: r["case"] != "clear"))})
        if not physical:
            tracking_rows, summary["tracking"] = tracking_benchmark(campus, search, args.output)
            publish(args.output / "tracking.csv", tracking_rows)
        report = json.dumps(summary, indent=2, allow_nan=False)
        (args.output / "summary.json").write_text(report + "\n", encoding="utf-8")
        print(report)
    except (OSError, ValueError, EOFError, cv2.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
