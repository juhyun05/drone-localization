"""Compare four matching modes on printed-map renders and optional XIAO frames."""

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import cv2
import numpy as np

from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.illumination_correction import IlluminationConfig
from localization.image_processing import ColorConfig
from localization.lighting_pipeline import LightingProcessor
from localization.map_loader import CampusMap, DEFAULT_MAP, MapRaster, hex_to_bgr
from localization.map_matching import MATCHING_MODES, MapMatcher, MatchCandidate, MatchDiagnostics, SearchConfig, angle_error
from localization.road_debug import road_panel, road_views
from localization.road_detection import detect_roads, direction_diversity
from localization.road_map import PRINT_PREVIEW, REFERENCE_PREVIEW
from localization.tracking_manager import TrackingManager
from tools.benchmark_environment import file_digest


OUTPUT = Path(__file__).parent / "experiments" / "hybrid_map_matching"
NOMINAL = CameraConfig(height_m=0.28)
SCENES = ("building_rich", "mixed", "intersection", "curved", "straight", "blank",
          "height_variation", "shadow", "occlusion")
POSITION_LIMIT_M = 0.05
YAW_LIMIT_DEG = 15


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"unable to encode {path.name}")
    data.tofile(path)


def camera(campus):
    result = SyntheticCamera(MapRaster(campus, 800), NOMINAL)
    path = PRINT_PREVIEW if PRINT_PREVIEW.is_file() else REFERENCE_PREVIEW
    result.scene = cv2.resize(read_image(path), result.raster.size, interpolation=cv2.INTER_LINEAR)
    return result


def synthetic_samples(sim, road_map, count, seed):
    # Assign scenes using the reference ink, independently of the camera detector.
    canvas = np.zeros(sim.scene.shape[:2], np.uint8)
    colors = {hex_to_bgr(road["display_color"]) for road in sim.raster.campus.roads}
    for color in colors:
        canvas |= cv2.inRange(sim.scene, color, color)
    buildings = sim.raster.labels()
    junctions = np.array(road_map.junctions_xy_m)
    size = NOMINAL.width_px, NOMINAL.height_px
    groups = {name: [] for name in SCENES}
    rng = np.random.default_rng(seed)
    for y in np.arange(0.23, 2.29, 0.072):
        for x in np.arange(0.23, 2.29, 0.072):
            pose = Pose(float(x), float(y), float(rng.integers(0, 360)))
            transform = sim.map_transform(pose)
            warp = lambda plane: cv2.warpAffine(plane, transform, size,
                                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            road = warp(canvas)
            building_fraction = float(np.count_nonzero(warp(buildings)) / road.size)
            road_fraction = float(np.count_nonzero(road) / road.size)
            diversity = direction_diversity(road)
            near_junction = bool(np.min(np.linalg.norm(junctions - (x, y), axis=1)) < 0.095)
            if building_fraction >= 0.23:
                groups["building_rich"].append(pose)
            elif 0.035 <= building_fraction <= 0.145 and road_fraction >= 0.08 and diversity >= 0.12:
                groups["mixed"].append(pose)
            elif building_fraction <= 0.005 and 0.052 <= road_fraction <= 0.46:
                if near_junction and diversity >= 0.17:
                    groups["intersection"].append(pose)
                elif not near_junction and diversity >= 0.17:
                    groups["curved"].append(pose)
                elif diversity <= 0.026:
                    groups["straight"].append(pose)
            elif building_fraction + road_fraction <= 0.001:
                groups["blank"].append(pose)
    chosen = {}
    for name, poses in groups.items():
        selected = []
        for index in rng.permutation(len(poses)):
            pose = poses[index]
            if any(math.hypot(pose.x_m - other.x_m, pose.y_m - other.y_m) < 0.15 for other in selected):
                continue
            selected.append(pose)
            if len(selected) >= count:
                break
        chosen[name] = selected
    road_poses = [p for pair in zip(chosen["intersection"], chosen["curved"]) for p in pair]
    if not road_poses:
        raise ValueError("reference has no intersection/curved test scenes")
    sim.config = NOMINAL
    for name in SCENES:
        poses = road_poses[:count] if name in ("height_variation", "shadow", "occlusion") else chosen[name]
        for index, pose in enumerate(poses):
            factor = (0.76, 1.00, 1.24, 1.39)[index % 4] if name == "height_variation" else 1
            sim.config = replace(NOMINAL, height_m=NOMINAL.height_m * factor)
            frame = sim.capture(pose)
            if name == "shadow":
                yy, xx = np.indices(frame.shape[:2])
                across = (xx + 0.47 * yy - 120) / 18
                field = 0.96 - 0.45 / (1 + np.exp(across))
                frame = np.rint(frame * field[:, :, None]).astype(np.uint8)
            if name == "occlusion":
                cv2.rectangle(frame, (98, 70), (143, 112), (85, 89, 92), -1)
            truth = MatchCandidate(1, pose.x_m, pose.y_m, pose.yaw_deg, sim.config.footprint_m)
            yield name, f"{index:02d}", frame, truth


def just_ms(start):
    return (perf_counter() - start) * 1000


def evaluate(dataset, samples, matchers, lighting, output):
    rows = []
    for scene, image_id, original, truth in samples:
        prepared = lighting.apply(original) if scene == "shadow" or dataset == "real" else None
        image = original if prepared is None else prepared.image
        confidence = None if prepared is None else prepared.confidence
        directory = output / ("synthetic" if dataset == "synthetic" else "real_camera/results") / scene / image_id
        save_image(directory / "original.png", original)
        for mode, matcher in matchers.items():
            for search in ("global", "local") if truth else ("global",):
                options = {}
                if search == "local":
                    # Shared, perturbed prior. No matching mode receives the true pose as a center.
                    options = dict(prior_xy=(truth.x_m + 0.018, truth.y_m - 0.016),
                                   prior_yaw=(truth.yaw_deg + 5) % 360,
                                   prior_span_m=truth.span_m * 1.025, radius_m=0.16)
                details = MatchDiagnostics() if search == "global" else None
                timings = {}
                labels, roads = [], []
                t0 = perf_counter()
                e = matcher.estimate(image, **options, confidence_mask=confidence, timings=timings,
                                     on_labels=labels.append, on_roads=roads.append)
                ms = just_ms(t0)
                if details is not None and mode == "adaptive":
                    matcher.estimate(image, confidence_mask=confidence, diagnostics=details)
                error = yaw_error = span_error = correct = false = None
                if truth:
                    correct = false = False
                    if e.success:
                        error = math.hypot(e.x_m - truth.x_m, e.y_m - truth.y_m)
                        yaw_error = angle_error(e.yaw_deg, truth.yaw_deg)
                        span_error = abs(e.span_m - truth.span_m)
                        correct = error <= POSITION_LIMIT_M and yaw_error <= YAW_LIMIT_DEG
                        false = not correct
                elif scene == "off_map":
                    false = e.success
                alt = details.runner_up if details else None
                row = dict(dataset=dataset, scene=scene, id=image_id, mode=mode, search=search, **asdict(e),
                           position_error_m=error, yaw_error_deg=yaw_error, span_error_m=span_error,
                           correct=correct, false_localization=false, ms=ms, hsv_ms=timings.get("hsv_ms"),
                           road_ms=timings.get("road_ms"), runner_up_score=alt.score if alt else None,
                           truth_x_m=truth.x_m if truth else None, truth_y_m=truth.y_m if truth else None,
                           truth_yaw_deg=truth.yaw_deg if truth else None, truth_span_m=truth.span_m if truth else None)
                rows.append(row)
                if mode == "adaptive" and search == "global":
                    road = roads[0] if roads else detect_roads(image, labels[0], confidence)
                    candidate = e if e.success else None
                    projection = (matcher.project_roads(image.shape, candidate) if candidate else
                                  np.zeros(image.shape[:2], np.uint8))
                    views = road_views(original, matcher.detector.label_palette, labels[0], road, projection)
                    for label, pixels in views.items():
                        if label != "Original":
                            save_image(directory / (label.lower().replace(" ", "_") + ".png"), pixels)
                    save_image(directory / "diagnostic.png", road_panel(views))
                    if details and details.candidates:
                        (directory / "candidates.json").write_text(json.dumps({
                            "accepted": e.success, "reason": e.reason,
                            "runner_up": asdict(alt) if alt else None,
                            "candidates": [asdict(c) for c in details.candidates],
                        }, indent=2), encoding="utf-8")
                    else:
                        (directory / "candidates.json").unlink(missing_ok=True)
                if truth is None and mode == "adaptive":
                    row["road_direction_diversity"] = roads[0].direction_diversity
        print(f"{dataset} {scene}/{image_id}", flush=True)
    return rows


def transition(sim, matchers):
    # Fixed after the pilot run so later revisions use the same route.
    start, end = np.array((0.734, 0.806)), np.array((1.454, 0.662))
    start_yaw, end_yaw = 177.0, 156.0
    steps = max(6, math.ceil(float(np.linalg.norm(end - start)) / 0.027))
    sim.config = NOMINAL
    yaw_delta = (end_yaw - start_yaw + 180) % 360 - 180
    seq = []
    for phase, fractions in (("building_to_road", np.linspace(0, 1, steps)),
                             ("road_hold", [1] * 4), ("blank", [1] * 11),
                             ("road_reappears", [1] * 5), ("road_to_building", np.linspace(1, 0, steps))):
        for fraction in fractions:
            xy = start + fraction * (end - start)
            yaw = (start_yaw + fraction * yaw_delta) % 360
            frame = (np.full((NOMINAL.height_px, NOMINAL.width_px, 3), 242, np.uint8)
                     if phase == "blank" else sim.capture(Pose(*xy, yaw)))
            seq.append((phase, xy, yaw, frame))
    results = []
    for mode, matcher in matchers.items():
        track = TrackingManager(matcher)
        for i, (phase, xy, yaw, image) in enumerate(seq):
            result = track.step(image, i * 0.2)
            observed = result.measurement if result.measurement_used else None
            correct = bool(observed and math.hypot(observed.x_m - xy[0], observed.y_m - xy[1]) <= POSITION_LIMIT_M
                           and angle_error(observed.yaw_deg, yaw) <= YAW_LIMIT_DEG)
            results.append(dict(mode=mode, phase=phase, frame=i, timestamp_s=result.timestamp_s,
                                state=result.state.value, attempt=result.attempt, source=result.observation_source,
                                observed=bool(observed), correct=correct, relocalized=result.relocalized,
                                reason=result.reason, sigma_m=result.sigma_m, match_ms=result.match_ms,
                                output_x_m=result.xy_m[0] if result.xy_m else None,
                                output_y_m=result.xy_m[1] if result.xy_m else None,
                                truth_x_m=float(xy[0]), truth_y_m=float(xy[1]), truth_yaw_deg=yaw))
    return results


def write_csv(path, rows, *, extras=()):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], *extras])
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    def metric(name, function):
        values = [r[name] for r in rows if r[name] is not None]
        return function(values) if values else None

    ms = [r["ms"] for r in rows]
    return dict(frames=len(rows), accepted=sum(r["success"] for r in rows),
                correct=metric("correct", sum), false=metric("false_localization", sum),
                false_evaluable_frames=sum(r["false_localization"] is not None for r in rows),
                rmse_m=metric("position_error_m", lambda a: math.sqrt(np.mean(np.square(a)))),
                yaw_mae_deg=metric("yaw_error_deg", lambda a: float(np.mean(a))),
                span_mae_m=metric("span_error_m", lambda a: float(np.mean(a))),
                score_median=metric("score", median), margin_median=metric("margin", median),
                median_ms=median(ms), p95_ms=float(np.percentile(ms, 95)), fps=1000 * len(ms) / sum(ms))


def summarize(rows, tracking):
    grouped = {}
    for row in rows:
        key = "/".join(row[k] for k in ("dataset", "scene", "mode", "search"))
        grouped.setdefault(key, []).append(row)
        key = "/".join((row["dataset"], "all", row["mode"], row["search"]))
        grouped.setdefault(key, []).append(row)
    track_summary = {}
    for mode in MATCHING_MODES:
        seq = [row for row in tracking if row["mode"] == mode]
        if not seq:
            continue
        t0 = next(row["timestamp_s"] for row in seq if row["phase"] == "road_reappears")
        recovered = next((row["timestamp_s"] for row in seq if row["timestamp_s"] >= t0 and row["correct"]), None)
        track_summary[mode] = dict(
            frames=len(seq), observed=sum(row["observed"] for row in seq), correct=sum(row["correct"] for row in seq),
            false=sum(row["observed"] and not row["correct"] for row in seq),
            tracking_losses=sum(prev["state"] != "RELOCALIZING" and row["state"] == "RELOCALIZING"
                                for prev, row in zip(seq, seq[1:])),
            prediction_only_s=0.2 * sum(row["source"] == "PREDICTION_ONLY" for row in seq),
            blank_observations=sum(row["observed"] for row in seq if row["phase"] == "blank"),
            blank_expired_frames=sum(row["output_x_m"] is None for row in seq if row["phase"] == "blank"),
            seconds_to_first_correct_after_blank=None if recovered is None else round(recovered - t0, 4))
    return dict(results={key: aggregate(values) for key, values in grouped.items()}, tracking=track_summary,
                opencv=cv2.__version__, opencv_threads=cv2.getNumThreads(),
                truth_limits=dict(position_m=POSITION_LIMIT_M, yaw_deg=YAW_LIMIT_DEG))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--samples-per-scene", type=int, default=5)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--real", action="append", default=[], metavar="SCENE=DIRECTORY")
    parser.add_argument("--real-only", action="store_true", help="update real results without changing synthetic data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    campus = CampusMap.load()
    lighting = LightingProcessor(IlluminationConfig("combined"), shadow_aware=True)
    search = SearchConfig(auto_scale=True, nominal_span_m=NOMINAL.footprint_m)
    matchers = {mode: MapMatcher(campus, replace(search, matching_mode=mode)) for mode in MATCHING_MODES}
    rows, tracking = [], []
    if not args.real_only:
        sim = camera(campus)
        roads = matchers["adaptive"].roads
        reference = matchers["adaptive"].raster
        for name, plane in (("region", roads.region), ("edges", roads.edges), ("centerline", roads.centerline)):
            save_image(args.output / "reference" / (name + ".png"), plane * 255)
        overlay = cv2.resize(sim.scene, reference.size, interpolation=cv2.INTER_AREA)
        overlay[roads.edges > 0] = (20, 195, 240)
        for xy in roads.junctions_xy_m:
            cv2.circle(overlay, tuple(np.rint(reference.xy_to_uv(xy)).astype(int)), 2, (230, 0, 140), -1)
        save_image(args.output / "reference" / "print_overlay.png", overlay)
        (args.output / "reference" / "junctions.json").write_text(json.dumps(roads.junctions_xy_m, indent=2))
        rows = evaluate("synthetic", synthetic_samples(sim, roads, args.samples_per_scene, args.seed),
                        matchers, lighting, args.output)
        write_csv(args.output / "comparison.csv", rows)
        for name, mode in (("baseline", "building"), ("road_only", "road"), ("fixed_hybrid", "fixed"),
                           ("adaptive_hybrid", "adaptive")):
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            write_csv(directory / "synthetic_comparison.csv", [row for row in rows if row["mode"] == mode])
        tracking = transition(sim, matchers)
        write_csv(args.output / "trajectory.csv", tracking)
    if args.real:
        config = PhysicalCameraConfig.load()
        preprocessor = CameraPreprocessor(config)
        search = replace(search, nominal_span_m=config.nominal_span_m or SearchConfig().nominal_span_m)
        color = ColorConfig(saturation_tolerance=78, hue_shift=3, hue_band_width=15)
        matchers = {mode: MapMatcher(campus, replace(search, matching_mode=mode), color) for mode in MATCHING_MODES}
        real_samples = []
        for item in args.real:
            scene, directory = item.split("=", 1)
            for path in sorted(Path(directory).iterdir()):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    real_samples.append((scene, path.stem, preprocessor.apply(read_image(path)), None))
        real_rows = evaluate("real", real_samples, matchers, lighting, args.output)
        write_csv(args.output / "real_camera" / "comparison.csv", real_rows, extras=("road_direction_diversity",))
        rows.extend(real_rows)
    summary = summarize(rows, tracking)
    print_path = PRINT_PREVIEW if PRINT_PREVIEW.is_file() else REFERENCE_PREVIEW
    summary["inputs"] = dict(seed=args.seed, samples_per_scene=args.samples_per_scene,
                             map_json_sha256=file_digest(DEFAULT_MAP),
                             reference_sha256=file_digest(print_path),
                             python=sys.version.split()[0], numpy=np.__version__)
    path = args.output / "performance_summary.json"
    if args.real_only and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous["results"] = {key: value for key, value in previous["results"].items() if not key.startswith("real/")}
        previous["results"].update(summary["results"])
        summary = previous
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
