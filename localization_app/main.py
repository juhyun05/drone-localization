"""Launch the interactive demo, save a snapshot, or run the evaluation suite."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path

from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.image_processing import ColorConfig
from localization.lighting_pipeline import add_lighting_arguments, configured_lighting
from localization.map_loader import CampusMap, DEFAULT_MAP, MapRaster
from localization.map_matching import MATCHING_MODES, MapMatcher, SearchConfig
from localization.paths import DATA_DIR, EXPERIMENTS_DIR
from localization.random_view import BlurConfig, RandomViewExperiment, summarize, write_json
from localization.scale_cli import add_scale_arguments, scale_arguments
from simulation.drone_trajectory import DroneTrajectory
from simulation.lighting import LIGHTING_CASES


def main(argv=None):
    camera_defaults = CameraConfig()
    search_defaults = SearchConfig()
    color_defaults = ColorConfig()
    blur_defaults = BlurConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true", help="run reproducible global tests")
    parser.add_argument("--save", type=Path, metavar="PNG", help="save a snapshot without opening a GUI")
    mode.add_argument("--track", choices=("line", "square", "circle", "waypoints"),
                      help="play a timestamped synthetic path")
    mode.add_argument("--benchmark-tracking", action="store_true", help="evaluate moving trajectories")
    mode.add_argument("--random-test", action="store_true", help="compare random original and blurred views")
    mode.add_argument("--camera-test", action="store_true", help="show XIAO stream without matching")
    mode.add_argument("--live-localization", action="store_true", help="localize a live XIAO stream")
    mode.add_argument("--video-file", type=Path, help="localize a saved video")
    parser.add_argument("--camera-url", help="HTTP stream URL printed by the firmware")
    parser.add_argument("--camera-config", type=Path, help="physical camera preprocessing YAML")
    parser.add_argument("--global-only", action="store_true", help="estimate each physical frame independently")
    parser.add_argument("--localization-mode", choices=("map", "apriltag", "fusion"),
                        help="tracking sources for live or replay input (default: fusion)")
    parser.add_argument("--map-version", choices=("original_map", "apriltag_map"),
                        help="physical layout on camera; defaults to the bundled tagged layout in fusion mode")
    parser.add_argument("--apriltag-map", type=Path, help="override the AprilTag layout JSON for apriltag_map")
    parser.add_argument("--min-tag-side-px", type=float, help="smallest usable raw-image tag side (default: 14 px)")
    parser.add_argument("--map-period-ms", type=float, default=0,
                        help="minimum interval between map searches while a usable tag supplies a tracking pose")
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument("--matching-mode", choices=MATCHING_MODES, default="building")
    matching.add_argument("--hybrid-matching", dest="matching_mode", action="store_const", const="adaptive",
                          help="use adaptive building and road matching")
    parser.add_argument("--road-debug", action="store_true", help="show observed and projected road layers")
    parser.add_argument("--ground-truth", type=float, nargs=2, metavar=("X", "Y"),
                        help="fixed physical camera position for error evaluation only")
    parser.add_argument("--result-age", type=float, default=5.0,
                        help="maximum PC receive age in seconds for a displayed localization result")
    parser.add_argument("--record", type=Path, nargs="?", metavar="AVI",
                        const=DATA_DIR / "recordings" / f"xiao_{datetime.now():%Y%m%d_%H%M%S_%f}.avi",
                        help="save live raw input and timestamps (default: a new AVI in data/recordings)")
    parser.add_argument("--localization-period-ms", type=float, default=0,
                        help="minimum start-to-start interval (0 runs without a limit)")
    parser.add_argument("--max-frame-age-ms", type=float, default=300,
                        help="reject frames older than this when localization starts")
    parser.add_argument("--gui-interval-ms", type=int, default=50, help="independent GUI update interval")
    parser.add_argument("--performance-monitor", action="store_true",
                        help="plot recent processing times and measure GUI render time")
    parser.add_argument("--save-performance-log", type=Path, nargs="?",
                        const=EXPERIMENTS_DIR / "frame_timing" / "performance_live.csv",
                        help="write a new per-frame performance CSV")
    parser.add_argument("--benchmark-live-pipeline", action="store_true",
                        help="compare 0/100/150/200 ms headlessly on live or recorded input")
    parser.add_argument("--benchmark-seconds", type=float, default=60,
                        help="measurement duration in seconds per mode (minimum 60)")
    parser.add_argument("--benchmark-warmup-seconds", type=float, default=0,
                        help="run the tracker before collecting benchmark samples")
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--benchmark-seed", type=int, help="seed the first mode order")
    parser.add_argument("--benchmark-periods", type=int, nargs="+", choices=(0, 100, 150, 200),
                        default=(0, 100, 150, 200))
    parser.add_argument("--opencv-threads", type=int,
                        help="override the process-wide OpenCV worker count")
    parser.add_argument("--benchmark-output", type=Path, default=EXPERIMENTS_DIR / "frame_timing")
    parser.add_argument("--benchmark-scenario", choices=("stationary", "moving"), default="stationary")
    parser.add_argument("--truth-yaw", type=float, help="known fixed heading for pipeline benchmark only")
    parser.add_argument("--blur-kernel", type=int, default=blur_defaults.kernel_size)
    parser.add_argument("--blur-sigma", type=float, default=blur_defaults.sigma)
    parser.add_argument("--min-building-coverage", type=float, default=0.05,
                        help="geometric building fraction for random view sampling; 0 includes empty views")
    parser.add_argument("--max-sampling-attempts", type=int, default=100)
    parser.add_argument("--random-count", type=int, help="run this many paired random views without a GUI")
    parser.add_argument("--random-json", type=Path, help="write random-view data in snapshot or batch mode")
    parser.add_argument("--tracking-csv", type=Path, help="write per-frame tracking benchmark results")
    parser.add_argument("--dt", type=float, default=0.25, help="capture interval in seconds")
    parser.add_argument("--speed", type=float, default=0.32, help="trajectory speed in m/s")
    parser.add_argument("--yaw-rate", type=float, default=12.0, help="camera yaw change in degrees/s")
    parser.add_argument("--waypoint", type=float, nargs=2, action="append", metavar=("X", "Y"))
    parser.add_argument("--initial-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="optional independent tracking position prior in meters")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--map-ppm", type=float, default=320)
    parser.add_argument("--x", type=float, default=1.73)
    parser.add_argument("--y", type=float, default=2.02)
    parser.add_argument("--yaw", type=float, default=37)
    parser.add_argument("--height", type=float, default=camera_defaults.height_m, help="simulated height in meters")
    parser.add_argument("--hfov", type=float, default=camera_defaults.hfov_deg, help="simulated horizontal FOV in degrees")
    parser.add_argument("--width", type=int, default=camera_defaults.width_px, help="frame width in pixels")
    parser.add_argument("--frame-height", type=int, default=camera_defaults.height_px, help="frame height in pixels")
    parser.add_argument("--noise", type=float, default=camera_defaults.noise_std, help="Gaussian noise standard deviation")
    parser.add_argument("--brightness", type=float, default=camera_defaults.brightness)
    parser.add_argument("--bgr-gains", type=float, nargs=3, default=camera_defaults.channel_gains)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nominal-span", type=float,
                        help="reference horizontal span in map meters")
    parser.add_argument("--scales", type=float, nargs="+", default=search_defaults.scale_factors)
    add_scale_arguments(parser)
    add_lighting_arguments(parser)
    parser.add_argument("--lighting-case", choices=LIGHTING_CASES,
                        help="apply a synthetic lighting condition to the random comparison view")
    parser.add_argument("--print-width-m", type=float,
                        help="measured width of the printed map square in meters, excluding margins")
    parser.add_argument("--match-ppm", type=float, default=search_defaults.pixels_per_m)
    parser.add_argument("--yaw-step", type=float, default=search_defaults.yaw_step_deg)
    parser.add_argument("--min-coverage", type=float, default=search_defaults.min_frame_fraction,
                        help="minimum detected building fraction of input pixels")
    parser.add_argument("--hue-tolerance", type=int, default=color_defaults.hue_tolerance)
    parser.add_argument("--hue-shift", type=int, default=color_defaults.hue_shift,
                        help="shift observed hue before palette lookup, in OpenCV hue units")
    parser.add_argument("--hue-band-width", type=int, default=color_defaults.hue_band_width,
                        help="group adjacent map hues when matching printed colors (0 uses individual colors)")
    parser.add_argument("--saturation-tolerance", type=int, default=color_defaults.saturation_tolerance)
    parser.add_argument("--value-tolerance", type=int, default=color_defaults.value_tolerance)
    args = parser.parse_args(argv)
    if args.opencv_threads is not None:
        if args.opencv_threads < 1:
            parser.error("--opencv-threads must be positive")
        import cv2
        cv2.setNumThreads(args.opencv_threads)
    physical = args.camera_test or args.live_localization or args.video_file
    localizing = args.live_localization or args.video_file
    if not localizing and any(value is not None for value in (
            args.localization_mode, args.map_version, args.apriltag_map, args.min_tag_side_px, args.map_period_ms or None)):
        parser.error("AprilTag and localization mode options require --live-localization or --video-file")
    if args.global_only and args.localization_mode not in (None, "map"):
        parser.error("--global-only requires map localization")
    if args.min_tag_side_px is not None and not (math.isfinite(args.min_tag_side_px) and args.min_tag_side_px > 0):
        parser.error("--min-tag-side-px must be positive and finite")
    if not math.isfinite(args.map_period_ms) or args.map_period_ms < 0:
        parser.error("--map-period-ms must be nonnegative and finite")
    args.localization_mode = args.localization_mode or ("map" if args.global_only else "fusion")
    if args.road_debug and args.localization_mode == "apriltag":
        parser.error("--road-debug requires map matching")
    if args.benchmark_live_pipeline:
        if not localizing or args.record or args.save_performance_log or args.performance_monitor:
            parser.error("pipeline benchmark requires live localization or video input and manages its own logs")
        if not math.isfinite(args.benchmark_seconds) or args.benchmark_seconds < 60:
            parser.error("--benchmark-seconds must be at least 60")
        if (not math.isfinite(args.benchmark_warmup_seconds) or args.benchmark_warmup_seconds < 0
                or args.benchmark_repeats < 1 or len(set(args.benchmark_periods)) != len(args.benchmark_periods)):
            parser.error("benchmark needs nonnegative warmup, positive repeats and distinct periods")
        if args.benchmark_scenario == "moving" and (args.ground_truth or args.truth_yaw is not None):
            parser.error("fixed ground truth applies only to the stationary benchmark")
    if args.truth_yaw is not None and (not args.benchmark_live_pipeline or not math.isfinite(args.truth_yaw)):
        parser.error("--truth-yaw must be finite and requires --benchmark-live-pipeline")
    if not localizing and (args.localization_period_ms != 0 or args.max_frame_age_ms != 300
                           or args.performance_monitor or args.save_performance_log):
        parser.error("localization timing options require --live-localization or --video-file")
    if not (math.isfinite(args.localization_period_ms) and args.localization_period_ms >= 0
            and math.isfinite(args.max_frame_age_ms) and args.max_frame_age_ms > 0
            and args.gui_interval_ms > 0):
        parser.error("period must be nonnegative and frame age and GUI interval must be positive")
    lighting = args.illumination_correction or args.illumination_method or args.shadow_aware or args.shadow_debug
    if args.road_debug and (args.matching_mode == "building" or not (args.live_localization or args.video_file)):
        parser.error("--road-debug requires a road matching mode with --live-localization or --video-file")
    if lighting and not (args.live_localization or args.video_file):
        parser.error("lighting correction and shadow diagnostics require --live-localization or --video-file")
    if args.lighting_case and not args.random_test:
        parser.error("--lighting-case requires --random-test")
    if physical:
        if not math.isfinite(args.result_age) or args.result_age <= 0:
            parser.error("--result-age must be positive and finite")
        if args.save or (args.video_file and (args.camera_url or args.record)):
            parser.error("use --camera-url and --record with live input; --save is for synthetic snapshots")
        if not args.video_file and not args.camera_url:
            parser.error("camera mode requires --camera-url")
        try:
            run_physical_camera(args, search_defaults)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        return
    if (args.camera_url or args.camera_config or args.record or args.global_only
            or args.ground_truth or args.print_width_m is not None):
        parser.error("camera options require --camera-test, --live-localization or --video-file")
    if args.save and (args.track or args.benchmark or args.benchmark_tracking):
        parser.error("--save is for single-frame and random-view snapshots")
    if (args.random_count is not None or args.random_json) and not args.random_test:
        parser.error("--random-count and --random-json require --random-test")
    if args.random_count is not None and (args.random_count < 1 or args.save):
        parser.error("--random-count must be positive and cannot be combined with --save")
    if args.random_json and not args.save and args.random_count is None:
        parser.error("--random-json requires --save or --random-count")

    try:
        campus = CampusMap.load(args.map)
        raster = MapRaster(campus, args.map_ppm)
        camera_config = CameraConfig(args.width, args.frame_height, args.height, args.hfov,
                                     args.noise, args.brightness, tuple(args.bgr_gains))
        span = search_defaults.nominal_span_m if args.nominal_span is None else args.nominal_span
        search = SearchConfig(pixels_per_m=args.match_ppm, nominal_span_m=span,
                              scale_factors=tuple(args.scales), yaw_step_deg=args.yaw_step,
                              min_frame_fraction=args.min_coverage, matching_mode=args.matching_mode,
                              **scale_arguments(args))
        colors = ColorConfig(args.hue_tolerance, args.saturation_tolerance, args.value_tolerance,
                             hue_shift=args.hue_shift, hue_band_width=args.hue_band_width)
        matcher = MapMatcher(campus, search, colors)
        def inside(x, y):
            return 0 <= x <= campus.width_m and 0 <= y <= campus.height_m

        if args.track:
            if args.track == "waypoints":
                if not args.waypoint or not all(inside(*xy) for xy in args.waypoint):
                    raise ValueError("provide --waypoint X Y points inside the map")
                trajectory = DroneTrajectory.waypoints(args.waypoint, args.speed)
            else:
                trajectory = DroneTrajectory.preset(args.track, args.speed)
            samples = trajectory.samples(args.dt, args.yaw, args.yaw_rate)
            if args.initial_xy is not None and not inside(*args.initial_xy):
                raise ValueError("initial position prior must be inside the map")
        elif args.benchmark_tracking:
            if not all(math.isfinite(value) and value > 0 for value in (args.speed, args.dt)):
                raise ValueError("trajectory speed and capture interval must be positive and finite")
        elif not (args.benchmark or args.random_test) and not inside(args.x, args.y):
            raise ValueError("initial camera position must be inside the map")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    if args.benchmark:
        from benchmark import run_benchmark

        run_benchmark(raster, matcher, camera_config)
        return
    if args.benchmark_tracking:
        from benchmark_tracking import run_tracking_benchmark
        run_tracking_benchmark(raster, matcher, camera_config, args.dt, args.speed,
                               args.tracking_csv, args.seed)
        return
    if args.random_test:
        try:
            run_random_view(args, raster, camera_config, matcher)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.map_viewer import MapViewer, TrackingViewer

    if args.track:
        viewer = TrackingViewer(raster, camera_config, matcher, samples, args.seed, args.initial_xy)
        plt.show()
        return

    camera = SyntheticCamera(raster, camera_config, seed=args.seed)
    viewer = MapViewer(camera, matcher, Pose(args.x, args.y, args.yaw), interactive=args.save is None)
    if args.save:
        viewer.figure.savefig(args.save, dpi=150)
        print(viewer.status.get_text())
    else:
        plt.show()


def run_physical_camera(args, search_defaults):
    from localization.apriltag_localization import AprilTagConfig, AprilTagLocalizer, DEFAULT_TAG_MAP, TagMap
    from localization.camera_interface import TimedVideoSource
    from localization.camera_preprocessing import (
        CameraPreprocessor, DEFAULT_CAMERA_CONFIG, PhysicalCameraConfig,
    )
    from localization.camera_recording import CameraRecorder
    from localization.camera_stream import XiaoCameraSource
    from localization.frame_scheduler import FrameScheduler, PerformanceLog, PerformanceStats
    from localization.live_localization import LiveLocalization
    from time import perf_counter

    config_path = args.camera_config or DEFAULT_CAMERA_CONFIG
    config = PhysicalCameraConfig.load(config_path)
    localizer = map_camera = campus = None
    note = ""
    effective_span = physical_per_map = None
    lighting = None
    tag_layout = tag_path = None
    map_version = args.map_version
    if not args.camera_test:
        campus = CampusMap.load(args.map)
        if map_version is None:
            uses_bundled_map = campus.source == DEFAULT_MAP.resolve() and args.localization_mode != "map"
            tagged = uses_bundled_map or args.localization_mode == "apriltag" or args.apriltag_map is not None
            map_version = "apriltag_map" if tagged else "original_map"
        if map_version == "apriltag_map":
            tag_path = args.apriltag_map or DEFAULT_TAG_MAP
            tag_layout = TagMap.load(campus, tag_path)
        elif args.apriltag_map is not None or args.localization_mode == "apriltag":
            raise ValueError("AprilTag layout requires --map-version apriltag_map")
        if args.ground_truth and not (0 <= args.ground_truth[0] <= campus.width_m
                                     and 0 <= args.ground_truth[1] <= campus.height_m):
            raise ValueError("--ground-truth must be inside the model map")
        span = args.nominal_span if args.nominal_span is not None else config.nominal_span_m
        geometry = "configured" if span is not None else "default reference"
        effective_span = span if span is not None else search_defaults.nominal_span_m
        if args.print_width_m is not None:
            if not math.isfinite(args.print_width_m) or args.print_width_m <= 0:
                raise ValueError("--print-width-m must be positive and finite")
            physical_per_map = args.print_width_m / campus.width_m
        search = replace(search_defaults, pixels_per_m=args.match_ppm, nominal_span_m=effective_span,
                         scale_factors=tuple(args.scales), yaw_step_deg=args.yaw_step,
                         min_frame_fraction=args.min_coverage, matching_mode=args.matching_mode,
                         **scale_arguments(args))
        colors = ColorConfig(args.hue_tolerance, args.saturation_tolerance, args.value_tolerance,
                             hue_shift=args.hue_shift, hue_band_width=args.hue_band_width)
        processor = configured_lighting(args)
        if processor.shadow_aware or processor.debug or processor.corrector.config.method != "none":
            lighting = processor
        def build_localizer():
            tag_config = AprilTagConfig()
            if args.min_tag_side_px is not None:
                tag_config = replace(tag_config, min_side_px=args.min_tag_side_px)
            tag_localizer = AprilTagLocalizer(tag_layout, physical_per_map, tag_config) if tag_layout else None
            return LiveLocalization(MapMatcher(campus, search, colors), CameraPreprocessor(config),
                                    global_only=args.global_only, lighting=lighting, road_debug=args.road_debug,
                                    localization_mode=args.localization_mode, tag_localizer=tag_localizer,
                                    map_period_ms=args.map_period_ms)

        lens = "on" if config.calibration_path else "off"
        scale_mode = "ON" if args.auto_scale else "OFF"
        note = (f"Auto-scale {scale_mode}  |  Ref. {effective_span:.3f} map m ({geometry})"
                f"\nLens correction {lens}  |  Match {args.matching_mode}  |  {map_version}")

    def build_source():
        return TimedVideoSource(args.video_file) if args.video_file else XiaoCameraSource(args.camera_url)

    if args.benchmark_live_pipeline:
        from tools.benchmark_environment import file_digest
        from tools.benchmark_live_pipeline import compare_modes

        input_config = dict(camera_config=file_digest(config_path), map=file_digest(args.map))
        if config.calibration_path:
            input_config["lens_calibration"] = file_digest(config.calibration_path)
        if tag_path:
            input_config["apriltag_layout"] = file_digest(tag_path)
        compare_modes(
            build_source, build_localizer, args.benchmark_output, seconds=args.benchmark_seconds,
            max_frame_age_ms=args.max_frame_age_ms, scenario=args.benchmark_scenario,
            input_kind="video_replay" if args.video_file else "live_mjpeg",
            input_label=str(args.video_file or args.camera_url),
            ground_truth=args.ground_truth, truth_yaw=args.truth_yaw,
            warmup_s=args.benchmark_warmup_seconds, repeats=args.benchmark_repeats,
            seed=args.benchmark_seed, periods=args.benchmark_periods,
            settings=dict(camera_config=str(config_path), map=str(args.map), settings_sha256=input_config,
                          search=asdict(search),
                          colors=asdict(colors), global_only=args.global_only,
                          localization_mode=args.localization_mode, map_version=map_version,
                          print_width_m=args.print_width_m, min_tag_side_px=args.min_tag_side_px,
                          map_period_ms=args.map_period_ms,
                          lighting=asdict(lighting.corrector.config) if lighting else None,
                          shadow_aware=bool(lighting and lighting.shadow_aware),
                          shadow_debug=bool(lighting and lighting.debug)),
        )
        return

    import matplotlib.pyplot as plt
    from visualization.live_camera_viewer import LiveCameraViewer

    if campus is not None:
        localizer = build_localizer()
        map_camera = SyntheticCamera(MapRaster(campus, args.map_ppm), CameraConfig())
    recorder = log = scheduler = viewer = None
    source = build_source()
    started = perf_counter()
    performance = PerformanceStats(sample_limit=600, ground_truth=args.ground_truth) if localizer else None

    def record_event(event):
        performance.add(event)
        if log is not None:
            log.write(event)

    try:
        if args.save_performance_log:
            log = PerformanceLog(args.save_performance_log)
        if args.record:
            recorder = CameraRecorder(args.record, {
                "camera_yaml": config_path.read_text(encoding="utf-8"),
                "mode": "camera-test" if args.camera_test else "localization",
                "localization_mode": args.localization_mode if localizer else None,
                "map_version": map_version,
                "map_period_ms": args.map_period_ms,
                "nominal_span_m": effective_span,
                "global_only": args.global_only,
                "print_width_m": args.print_width_m,
                "scale_search": scale_arguments(args),
                "illumination": asdict(lighting.corrector.config) if lighting else None,
                "shadow_aware": bool(lighting and lighting.shadow_aware),
                "localization_period_ms": args.localization_period_ms,
                "max_frame_age_ms": args.max_frame_age_ms,
            })
        scheduler = FrameScheduler(source, localizer.process if localizer else None,
                                   period_ms=args.localization_period_ms,
                                   max_frame_age_ms=args.max_frame_age_ms,
                                   on_event=record_event if performance is not None else None,
                                   recorder=recorder)
        viewer = LiveCameraViewer(
            source, localizer, map_camera, scheduler=scheduler, performance=performance,
            performance_monitor=args.performance_monitor, gui_interval_ms=args.gui_interval_ms,
            geometry_note=note, ground_truth=args.ground_truth, result_age_s=args.result_age,
            nominal_span_m=effective_span, physical_per_map=physical_per_map,
        )
        started = perf_counter()
        source.start()
        scheduler.start()
        plt.show()
    finally:
        elapsed = max(perf_counter() - started, 1e-9)
        if viewer is not None:
            viewer.close()
        if scheduler is not None:
            scheduler.stop.set()
        source.close()
        if scheduler is not None:
            scheduler.close()
        if recorder is not None:
            recorder.close()
        if log is not None:
            log.close()
        if viewer is not None and (args.performance_monitor or args.save_performance_log):
            received = source.received if args.video_file else source.status().received
            print(json.dumps(performance.summary(received, elapsed), indent=2))


def run_random_view(args, raster, config, matcher):
    blur = BlurConfig(args.blur_kernel, args.blur_sigma)
    experiment = RandomViewExperiment(raster, config, matcher, seed=args.seed,
                                      min_building_coverage=args.min_building_coverage,
                                      max_attempts=args.max_sampling_attempts,
                                      lighting_case=args.lighting_case)
    if args.random_count is not None:
        original, blurred, records = [], [], []
        for index in range(args.random_count):
            try:
                result = experiment.run(blur)
            except ValueError as exc:
                raise ValueError(f"batch stopped after {index}/{args.random_count} pairs: {exc}") from exc
            original.append(result.original)
            blurred.append(result.blurred)
            if args.random_json:
                records.append(result.record())
            if (index + 1) % 10 == 0:
                print(f"Evaluated {index + 1}/{args.random_count} pairs", flush=True)
        summary = {"original": summarize(original), "blurred": summarize(blurred)}
        if args.random_json:
            write_json(args.random_json, {"summary": summary, "samples": records})
        print(json.dumps(summary, indent=2))
        return

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tests.random_view_test import RandomViewViewer

    viewer = RandomViewViewer(experiment, blur, interactive=args.save is None)
    if args.save:
        result = experiment.run(blur)
        viewer.show_result(result)
        viewer.save(args.save, args.random_json)
        print(json.dumps(result.record(), indent=2))
    else:
        plt.show()


if __name__ == "__main__":
    main()
