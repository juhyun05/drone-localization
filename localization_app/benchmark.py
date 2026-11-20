"""Reproducible, pose-blind evaluation of global localization on the campus map."""

from dataclasses import replace
import math
from time import perf_counter

import cv2
import numpy as np

from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera, image_center, perturb
from localization.map_loader import MapRaster
from localization.map_matching import MapMatcher, angle_error


def projected_frame(campus, pose: Pose, config: CameraConfig, seed: int) -> np.ndarray:
    """Draw vector features directly into a supersampled camera view, without a map crop."""
    width, height = config.width_px * 3, config.height_px * 3
    pixels_per_m = width / config.footprint_m
    yaw = math.radians(pose.yaw_deg)
    right = np.array([math.cos(yaw), math.sin(yaw)])
    down = np.array([math.sin(yaw), -math.cos(yaw)])

    def to_camera_pixels(points):
        offsets = np.asarray(points) - (pose.x_m, pose.y_m)
        uv = np.column_stack((offsets @ right, offsets @ down))
        return uv * pixels_per_m + image_center(width, height)

    image = campus.paint((width, height), to_camera_pixels, pixels_per_m)
    image = cv2.resize(image, (config.width_px, config.height_px), interpolation=cv2.INTER_AREA)
    return perturb(image, config, np.random.default_rng(seed))


def run_benchmark(raster: MapRaster, matcher: MapMatcher, config: CameraConfig) -> None:
    """Print per-frame errors and aggregate metrics for accepted and rejected poses."""
    clean = replace(config, noise_std=0, brightness=1, channel_gains=(1, 1, 1))
    camera = SyntheticCamera(raster, clean)
    sites = [
        Pose(1.73, 2.02, 37), Pose(0.83, 0.82, 124), Pose(2.29, 1.76, 283),
        Pose(0.49, 2.26, 348), Pose(1.53, 0.77, 67),
    ]
    variants = [
        ("crop", [(replace(p, yaw_deg=0), clean) for p in sites]),
        ("rotated", [(p, clean) for p in sites]),
        ("scaled", [(p, replace(clean, height_m=clean.height_m * f))
                    for p, f in zip(sites, (0.83, 1.16, 0.91, 1.07, 1.19))]),
        ("noise", [(p, replace(clean, noise_std=4)) for p in sites]),
        ("color", [(p, replace(clean, brightness=0.83, channel_gains=(1.02, 0.99, 1.01)))
                   for p in sites]),
        ("combined", [(p, replace(clean, height_m=clean.height_m * f, noise_std=4,
                                 brightness=0.87, channel_gains=(1.02, 0.99, 1.01)))
                      for p, f in zip(sites, (1.15, 0.88, 1.06, 0.81, 1.17))]),
        ("sparse", [(Pose(1.30, 0.23, 12), clean), (Pose(0.95, 1.71, 198), clean),
                    (Pose(1.54, 0.51, 72), clean)]),
    ]
    all_rows = []
    print("group      truth (m,m,deg)    error (m,deg)   ms     IoU/margin   outcome")
    for group, cases in variants:
        rows = []
        for i, (pose, settings) in enumerate(cases):
            frame = (camera.capture(pose) if group == "crop" else
                     projected_frame(raster.campus, pose, settings, 42 + i))
            start = perf_counter()
            result = matcher.estimate(frame)
            elapsed = (perf_counter() - start) * 1000
            error_m = yaw_error = None
            if result.success:
                error_m = math.hypot(pose.x_m - result.x_m, pose.y_m - result.y_m)
                yaw_error = angle_error(pose.yaw_deg, result.yaw_deg)
            errors = f"{error_m:.4f}, {yaw_error:5.1f}" if result.success else "    --,    --"
            print(f"{group:10} ({pose.x_m:.2f}, {pose.y_m:.2f}, {pose.yaw_deg:3.0f})  "
                  f"{errors}  {elapsed:6.1f}  {result.score:.3f}/{result.margin:.3f}  {result.reason}")
            rows.append((error_m, yaw_error, elapsed))
        _print_summary(group, rows)
        all_rows.extend(rows)
    _print_summary("all", all_rows)


def _print_summary(name, rows):
    accepted = [(position, yaw) for position, yaw, _ in rows if position is not None]
    correct = sum(position <= 0.05 and yaw <= 10 for position, yaw in accepted)
    avg_ms = np.mean([elapsed for _, _, elapsed in rows])
    errors = "RMSE=n/a, mean yaw=n/a"
    if accepted:
        positions, angles = np.asarray(accepted).T
        errors = (f"RMSE={np.sqrt(np.mean(positions ** 2)):.4f} m, "
                  f"mean yaw={np.mean(angles):.2f} deg")
    print(f"{name}: accepted {len(accepted)}/{len(rows)}, "
          f"within 5 cm and 10 deg {correct}/{len(rows)}, {errors}, mean {avg_ms:.1f} ms")
