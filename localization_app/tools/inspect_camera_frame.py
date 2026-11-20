"""Save a camera frame, HSV mask and an independent global matching result."""

import argparse
from dataclasses import asdict, replace
import csv
import json
import math
from pathlib import Path
import sys
from time import monotonic, perf_counter

import cv2
import numpy as np
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization.camera_preprocessing import (
    CameraPreprocessor, DEFAULT_CAMERA_CONFIG, PhysicalCameraConfig,
)
from localization.camera_stream import XiaoCameraSource
from localization.image_processing import ColorConfig
from localization.lighting_pipeline import add_lighting_arguments, configured_lighting, diagnostic_panel
from localization.map_loader import CampusMap
from localization.map_matching import MatchDiagnostics, MapMatcher, SearchConfig, angle_error
from localization.scale_cli import add_scale_arguments, scale_arguments
from tools.calibrate_camera import read_image, write_image


def capture_frame(url):
    source = XiaoCameraSource(url)
    source.start()
    try:
        deadline = monotonic() + 12
        while monotonic() < deadline:
            frame = source.read(timeout=0.25)
            if frame is not None:
                return frame.image
            if source.status().state == "failed":
                break
        raise ValueError(f"no camera frame: {source.status().error or 'timed out'}")
    finally:
        source.close()


def hsv_stats(pixels):
    hue_counts = np.bincount(pixels[:, 0], minlength=180)
    return {
        "pixels": len(pixels),
        "dominant_hue": int(hue_counts.argmax()),
        "hue_histogram": hue_counts.tolist(),
        "sv_p10_p50_p90": np.percentile(pixels[:, 1:], [10, 50, 90], axis=0).tolist(),
    }


def inspect_frame(frame, matcher, preprocessor, output, ground_truth=None, truth_yaw=None, roi=None,
                  *, debug_scale=False, lighting=None):
    """Write reproducible per-frame diagnostics. Truth values affect only error metrics."""
    pre_start = perf_counter()
    original = preprocessor.apply(frame)
    lighting_result = lighting.apply(original) if lighting is not None else None
    image = lighting_result.image if lighting_result is not None else original
    preprocess_ms = (perf_counter() - pre_start) * 1000
    labels = matcher.detector.detect(image, include_contours=False).labels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    palette = ((0, 0, 0), *matcher.raster.campus.palette)
    mask = np.asarray(palette, np.uint8)[labels]
    colors = {str(int(label)): hsv_stats(hsv[labels == label])
              for label in np.unique(labels) if label}
    roi_result = None
    if roi:
        x, y, w, h = roi
        if not (0 <= x < x + w <= image.shape[1] and 0 <= y < y + h <= image.shape[0]):
            raise ValueError("--hsv-roi must be inside the preprocessed image")
        roi_result = hsv_stats(hsv[y:y + h, x:x + w].reshape(-1, 3))
    started = perf_counter()
    diagnostics = MatchDiagnostics() if debug_scale else None
    confidence = lighting_result.confidence if lighting_result is not None else None
    estimate = matcher.estimate(image, diagnostics=diagnostics, confidence_mask=confidence)
    match_ms = (perf_counter() - started) * 1000
    row = dict(asdict(estimate), match_ms=match_ms,
               building_fraction=np.count_nonzero(labels) / labels.size,
               true_x_m=ground_truth[0] if ground_truth else None,
               true_y_m=ground_truth[1] if ground_truth else None,
               truth_yaw_deg=truth_yaw, error_cm=None, yaw_error_deg=None)
    if lighting_result is not None:
        shadow = lighting_result.shadow
        row.update(preprocess_ms=preprocess_ms, illumination=lighting.corrector.config.method,
                   shadow_aware=lighting.shadow_aware,
                   effective_coverage=shadow.effective_coverage if confidence is not None else 1.0,
                   shadow_ratio=shadow.shadow_ratio if shadow is not None else None)
    if estimate.success:
        if ground_truth:
            row["error_cm"] = 100 * math.dist((estimate.x_m, estimate.y_m), ground_truth)
        if truth_yaw is not None:
            row["yaw_error_deg"] = angle_error(estimate.yaw_deg, truth_yaw)

    output.mkdir(parents=True, exist_ok=False)
    write_image(output / "input.png", frame)
    write_image(output / "preprocessed.png", original)
    if lighting_result is not None:
        write_image(output / "illumination_corrected.png", image)
        if lighting.debug:
            write_image(output / "shadow_mask.png", shadow.shadow_mask.astype(np.uint8) * 255)
            write_image(output / "confidence.png", np.rint(shadow.confidence_mask * 255).astype(np.uint8))
            write_image(output / "lighting_comparison.png",
                        diagnostic_panel(original, lighting_result, matcher.detector, palette[1:]))
    write_image(output / "building_mask.png", mask)
    write_image(output / "comparison.png", np.hstack((image, mask)))
    if diagnostics is not None:
        write_scale_debug(output, matcher, labels, diagnostics)
    with (output / "localization_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    (output / "hsv_distribution.json").write_text(
        json.dumps({"classified_labels": colors, "roi_xywh": roi, "roi": roi_result}, indent=2),
        encoding="utf-8")
    return row


def write_scale_debug(output, matcher, labels, diagnostics):
    stages = {"coarse": diagnostics.coarse_by_span, "verified": diagnostics.verified_by_span}
    scores = {stage: [asdict(peaks[span]) for span in sorted(peaks)]
              for stage, peaks in stages.items()}
    data = dict(by_stage=scores, top_candidates=[asdict(c) for c in diagnostics.candidates],
                decision_runner_up=asdict(diagnostics.runner_up) if diagnostics.runner_up else None)
    (output / "scale_search.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    if not diagnostics.candidates:
        return

    # The highest-scoring hypothesis is useful even when the margin check rejected it.
    candidate = diagnostics.candidates[0]
    predicted = matcher.project_labels(labels.shape, candidate)
    palette = np.asarray(((0, 0, 0), *matcher.raster.campus.palette), np.uint8)
    write_image(output / "mask_alignment.png", np.hstack((palette[labels], palette[predicted])))

    from matplotlib.figure import Figure

    figure = Figure(figsize=(6, 3.4), layout="constrained")
    axes = figure.subplots()
    for stage, peaks in stages.items():
        if peaks:
            widths = sorted(peaks)
            axes.plot(widths, [peaks[span].score for span in widths], ".-", label=stage)
    axes.set(xlabel="Horizontal span (map m)", ylabel="Peak color IoU", ylim=(0, 1))
    axes.legend()
    figure.savefig(output / "scale_scores.png", dpi=125)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--url")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--hsv-settings", type=Path, help="reload saved HSV tolerance YAML")
    parser.add_argument("--hsv-tolerances", type=int, nargs=3, metavar=("H", "S", "V"))
    parser.add_argument("--hsv-roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--nominal-span", type=float, help="ground width in map meters")
    add_scale_arguments(parser)
    add_lighting_arguments(parser)
    parser.add_argument("--debug-scale", action="store_true", help="save scale scores and alignment")
    parser.add_argument("--ground-truth", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--truth-yaw", type=float)
    args = parser.parse_args()
    try:
        camera = PhysicalCameraConfig.load(args.camera_config)
        colors = ColorConfig()
        if args.hsv_settings:
            settings = yaml.safe_load(args.hsv_settings.read_text(encoding="utf-8"))
            colors = ColorConfig(**settings)
        if args.hsv_tolerances:
            h, s, v = args.hsv_tolerances
            colors = replace(colors, hue_tolerance=h, saturation_tolerance=s, value_tolerance=v)
        search = SearchConfig(**scale_arguments(args))
        span = args.nominal_span if args.nominal_span is not None else camera.nominal_span_m
        if span is not None:
            search = replace(search, nominal_span_m=span)
        else:
            print("Using the default reference span in map meters.")
        matcher = MapMatcher(CampusMap.load(), search, colors)
        frame = read_image(args.image) if args.image else capture_frame(args.url)
        lighting = configured_lighting(args)
        if not (lighting.debug or lighting.shadow_aware or lighting.corrector.config.method != "none"):
            lighting = None
        row = inspect_frame(frame, matcher, CameraPreprocessor(camera), args.output,
                            args.ground_truth, args.truth_yaw, args.hsv_roi,
                            debug_scale=args.debug_scale, lighting=lighting)
        (args.output / "hsv_settings.yaml").write_text(
            yaml.safe_dump(asdict(colors)), encoding="utf-8")
        (args.output / "search_settings.json").write_text(
            json.dumps(asdict(search), indent=2), encoding="utf-8")
        camera_settings = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
        if camera.calibration_path:
            camera_settings["calibration"]["calibration_file"] = str(camera.calibration_path.resolve())
        (args.output / "camera_settings.yaml").write_text(
            yaml.safe_dump(camera_settings, sort_keys=False), encoding="utf-8")
        print(json.dumps(row, indent=2))
    except (OSError, ValueError, KeyError, TypeError, cv2.error, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
