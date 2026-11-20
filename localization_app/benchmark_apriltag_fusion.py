"""Compare map, AprilTag and fusion on the same frames with no GUI or network."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from localization.apriltag_detector import detect_tags, make_tag_detector
from localization.apriltag_localization import AprilTagLocalizer, TagMap
from localization.camera_interface import CameraFrame, TimedVideoSource, VideoFileSource
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.frame_scheduler import FrameScheduler, PerformanceStats
from localization.illumination_correction import IlluminationConfig
from localization.image_processing import ColorConfig
from localization.lighting_pipeline import LightingProcessor
from localization.live_localization import LiveLocalization
from localization.localization_fusion import yaw_difference
from localization.map_loader import BACKGROUND, CampusMap, MapRaster
from localization.map_matching import MapMatcher, SearchConfig
from localization.paths import DATA_DIR


MODES = ("map", "apriltag", "fusion")
SIZE = (320, 240)
MATRIX = np.array([[277., 0., 159.5], [0., 277., 119.5], [0., 0., 1.]])
DISTORTION = np.zeros(5)


@dataclass(frozen=True)
class Sample:
    image: np.ndarray
    case: str
    timestamp: float
    xy: tuple | None = None
    yaw: float | None = None
    tag_present: bool | None = None
    span: float | None = None


def read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def synthetic_samples(campus, layout):
    raster = MapRaster(campus, 560)
    reference = raster.render()
    map_corners = np.array([[0., 2.5], [2.5, 2.5], [2.5, 0.], [0., 0.]])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    markers = {i: cv2.aruco.generateImageMarker(dictionary, i, 120) for i in (*layout.tags, 100)}
    patch_corners = np.array([[0., 0.], [119., 0.], [119., 119.], [0., 119.]], np.float32)

    def render(case, t, xy, yaw, height=.43, tilt=0., building=True, tag_ids=(), degraded=False, outlier=None):
        angle = np.radians(yaw)
        c, s = np.cos(angle), np.sin(angle)
        level = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.]])
        rotation = cv2.Rodrigues(np.array([np.radians(tilt), 0., 0.]))[0] @ level
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = -rotation @ np.r_[xy, height]

        def project(points):
            uv, _ = cv2.projectPoints(np.c_[points, np.zeros(len(points))], rvec, tvec, MATRIX, DISTORTION)
            return uv.reshape(-1, 2).astype(np.float32)

        if building:
            homography = cv2.getPerspectiveTransform(raster.xy_to_uv(map_corners).astype(np.float32),
                                                     project(map_corners))
            image = cv2.warpPerspective(reference, homography, SIZE, borderValue=BACKGROUND)
        else:
            image = np.full((*SIZE[::-1], 3), BACKGROUND, np.uint8)
        visible = False
        for tag_id in tag_ids:
            tag = layout.tags[tag_id]
            corners = project(tag.corners)
            paper = project(tag.paper_corners)
            if np.any(paper.max(axis=0) < 0) or np.any(paper.min(axis=0) >= SIZE):
                continue
            visible |= bool(np.all(corners >= 0) and np.all(corners < SIZE))
            cv2.fillConvexPoly(image, np.rint(paper).astype(np.int32), (255, 255, 255))
            transform = cv2.getPerspectiveTransform(patch_corners, corners)
            marker = markers[100 if outlier == tag_id else tag_id]
            rendered = cv2.warpPerspective(marker, transform, SIZE, flags=cv2.INTER_NEAREST, borderValue=255)
            inside = np.zeros(SIZE[::-1], np.uint8)
            cv2.fillConvexPoly(inside, np.rint(corners).astype(np.int32), 1)
            ink = (inside > 0) & (rendered == 0)
            image[ink] = 0
        if degraded:
            shadow = np.linspace(.54, 1., SIZE[0]).reshape(1, -1, 1)
            image = cv2.GaussianBlur(np.rint(image * shadow).astype(np.uint8), (5, 5), .9)
        span = (SIZE[0] - 1) * height / MATRIX[0, 0] if tilt == 0 else None
        return Sample(image, case, t, xy, yaw, visible, span)

    cluster = (20, 21, 22, 23, 27, 38)
    for i in range(4):
        t = i * .2
        yield render("building_only", t, (2.17 + i * .006, 1.85), 15, height=.52)
    for i in range(4):
        yield render("tag_only", i * .2, (.50, .69), 358, height=.28, tilt=10,
                     building=False, tag_ids=(20,))
    for i in range(5):
        yield render("building_and_tags", i * .2, (2.19, 1.83 + i * .006), 19, height=.51,
                     tag_ids=(7, 8, 9, 10, 12))
    for i in range(4):
        yield render("multi_tag", i * .2, (.66, .63), 6, building=False, tag_ids=cluster)
    for i in range(4):
        yield render("no_features", i * .2, (.66, .63), 6, building=False)
    for i in range(11):
        xy = (.66, .63) if i < 6 else (2.19, 1.84)
        tags = cluster if i < 2 else (7, 8, 9, 10, 12) if i >= 6 else ()
        yield render("tag_loss_and_reappearance", i * .2, xy, 6, height=.52, building=False, tag_ids=tags)
    for i, height in enumerate((.33, .39, .45, .51, .58, .49)):
        yield render("height_change", i * .2, (.69, .65), 9, height=height, tag_ids=cluster)
    for i, tilt in enumerate((4., 8., 12., 17., 21.)):
        yield render("tilt", i * .2, (.69, .65), 11, tilt=tilt, tag_ids=cluster)
    for i in range(4):
        yield render("shadow_blur", i * .2, (2.19, 1.83), 19, height=.51,
                     tag_ids=(7, 8, 9, 10, 12), degraded=True)
    for i in range(4):
        yield render("outlier_tag", i * .2, (.66, .63), 6, building=False, tag_ids=cluster, outlier=21)


def run_paired(samples, builder):
    output = {}
    for mode in MODES:
        case = localizer = None
        rows = []
        for index, sample in enumerate(samples, 1):
            if case != sample.case:
                localizer = builder(mode)
                case = sample.case
            frame = CameraFrame(sample.image, sample.timestamp, index, case)
            result = localizer.process(frame)
            track = result.tracking
            obs = track.correction
            error = np.linalg.norm(np.subtract(obs.xy, sample.xy)) if obs and sample.xy else None
            yaw = abs(yaw_difference(obs.yaw_deg, sample.yaw)) if obs and sample.yaw is not None else None
            tracked_error = np.linalg.norm(np.subtract(track.xy_m, sample.xy)) if track.xy_m and sample.xy else None
            row = dict(case=case, timestamp=sample.timestamp, ground_truth=sample.xy, source=track.observation_source,
                       state=track.state.value, reason=track.reason, accepted=obs is not None,
                       relocalized=track.relocalized, tag_expected=sample.tag_present,
                       detected=bool(result.tag_detections), position_error_m=error, yaw_error_deg=yaw,
                       tracking_error_m=tracked_error, processing_ms=result.processing_ms,
                       detection_ms=result.tag_detection_ms, pose_ms=result.tag_pose_ms,
                       matching_ms=result.matching_ms, filter_ms=result.tracking_ms, fusion_ms=result.fusion_ms)
            if obs and sample.span is not None and obs.span_m is not None:
                row["span_error_m"] = abs(sample.span - obs.span_m)
            rows.append(row)
        grouping = defaultdict(list)
        for row in rows:
            grouping[row["case"]].append(row)
        output[mode] = dict(overall=summarize(rows), cases={k: summarize(v) for k, v in grouping.items()})
    return output


def summarize(rows):
    def values(key):
        return [r[key] for r in rows if r.get(key) is not None]

    def distribution(key):
        v = values(key)
        return dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95))) if v else None

    pos = values("position_error_m")
    track_pos = values("tracking_error_m")
    accepted = sum(values("accepted"))
    wrong = sum(r["position_error_m"] is not None and (r["position_error_m"] > .05 or r["yaw_error_deg"] > 15)
                for r in rows)
    loss = 0
    recovery_times = []
    previous = lost_at = None
    for row in rows:
        if previous is None or row["case"] != previous["case"]:
            lost_at = None
        if row["state"] == "RELOCALIZING" and lost_at is None:
            loss += 1
            lost_at = row["timestamp"]
        if row["relocalized"] and lost_at is not None:
            recovery_times.append(row["timestamp"] - lost_at)
            lost_at = None
        previous = row
    timings = {k: distribution(k + "_ms") for k in ("processing", "detection", "pose", "matching", "filter", "fusion")}
    absent = [r for r in rows if r["tag_expected"] is False]
    truth = any(r["ground_truth"] is not None for r in rows)
    return dict(frames=len(rows), accepted=accepted, accepted_rate=accepted / len(rows),
                success_rate=(accepted - wrong) / len(rows) if truth else None,
                false_localization_rate=wrong / accepted if pos else None,
                position_rmse_m=float(np.sqrt(np.mean(np.square(pos)))) if pos else None,
                tracked_rmse_m=float(np.sqrt(np.mean(np.square(track_pos)))) if track_pos else None,
                yaw_error_deg=distribution("yaw_error_deg"), span_error_m=distribution("span_error_m"),
                tracking_loss_count=loss, relocalization_s=recovery_times,
                tag_detected_frames=sum(values("detected")), no_tag_frames=len(absent),
                no_tag_accepted=sum(r["accepted"] for r in absent),
                no_tag_predictions=sum(r["source"] == "PREDICTION_ONLY" for r in absent),
                source_counts=Counter(row["source"] or "UNLOCALIZED" for row in rows), timing_ms=timings,
                offline_processing_fps=1000 / timings["processing"]["mean"])


def sequential_video(path, count):
    video = VideoFileSource(path)
    try:
        for _ in range(count):
            frame = video.read()
            if frame is None:
                break
            yield Sample(frame.image, "recording", frame.timestamp)
    finally:
        video.close()


def scheduled_video(path, mode, builder, seconds):
    source = TimedVideoSource(path)
    stats = PerformanceStats()
    process = builder(mode).process
    start = perf_counter()
    scheduler = FrameScheduler(source, process, deadline=start + seconds, on_event=stats.add)
    source.start()
    scheduler.start()
    try:
        scheduler.done.wait(seconds + 15)
    finally:
        source.close()
        scheduler.close()
    if source.error or scheduler.error:
        raise ValueError(source.error or scheduler.error)
    return stats.summary(source.received, perf_counter() - start)


def evaluate_donor_frames(directory, layout):
    truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))
    intrinsics = truth["camera"]
    raw_size = intrinsics["width"], intrinsics["height"]
    detector = make_tag_detector(layout.family)
    solver = AprilTagLocalizer(layout, 1.)
    output = {}
    for size in (raw_size, SIZE):
        calibration = np.array(intrinsics["K"])
        scale = np.asarray(size) / raw_size
        calibration[:2, :2] *= scale[:, None]
        calibration[:2, 2] = (calibration[:2, 2] + .5) * scale - .5
        disposition = Counter()
        positions, turns = [], []
        for i, frame in enumerate(truth["frames"], 1):
            image = read_image(directory / frame["file"])
            if size != raw_size:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            detections = detect_tags(image, detector, layout.family, i, float(i))
            observation = solver.solve(detections, size, (calibration, DISTORTION))
            disposition["frames"] += 1
            disposition["frames_with_visible_tag"] += bool(frame["tags_fully_visible"])
            if observation is None:
                disposition["no detection"] += 1
            elif not observation.accepted:
                disposition[observation.rejection_reason] += 1
            else:
                disposition["accepted"] += 1
                positions.append(float(np.linalg.norm(np.subtract(observation.xy, frame["camera_xy_m"]))))
                turns.append(abs(yaw_difference(observation.yaw_deg, -frame["heading_deg"])))
        output[f"{size[0]}x{size[1]}"] = dict(counts=disposition,
                                             rmse_m=float(np.sqrt(np.mean(np.square(positions)))) if positions else None,
                                             mean_yaw_error_deg=float(np.mean(turns)) if turns else None)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="optional existing recording, evaluated without ground truth")
    parser.add_argument("--video-frames", type=int, default=30)
    parser.add_argument("--scheduled-seconds", type=float, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--camera-config", type=Path, help="camera YAML for recorded input")
    parser.add_argument("--print-width-m", type=float, help="measured map width for recorded input")
    parser.add_argument("--a4-color-profile", action="store_true", help="use the existing XIAO A4 HSV settings on replay")
    parser.add_argument("--donor-frames", type=Path, nargs="?", metavar="DIRECTORY",
                        const=DATA_DIR / "experiments" / "incident_camera" / "sim_frames",
                        help="validate poses on the bundled simulated frames or on a supplied directory")
    parser.add_argument("--map-period-ms", type=float, default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.video_frames < 1 or not np.isfinite(args.scheduled_seconds) or args.scheduled_seconds < 0:
        parser.error("need positive thread and frame counts and a finite, nonnegative duration")
    if (not np.isfinite(args.map_period_ms) or args.map_period_ms < 0
            or (args.print_width_m is not None and not (np.isfinite(args.print_width_m) and args.print_width_m > 0))):
        parser.error("map interval must be nonnegative and measured print width must be positive and finite")
    cv2.setNumThreads(args.threads)
    campus = CampusMap.load()
    layout = TagMap.load(campus)
    config = PhysicalCameraConfig.load()
    search = SearchConfig(auto_scale=True, nominal_span_m=.53, matching_mode="adaptive")
    with TemporaryDirectory() as directory:
        calibration = Path(directory) / "synthetic.npz"
        np.savez(calibration, camera_matrix=MATRIX, dist_coeffs=DISTORTION, image_size=SIZE)

        def make_builder(preprocess_config, scale, *, real_a4=False):
            colors = (ColorConfig(saturation_tolerance=78, hue_shift=3, hue_band_width=15)
                      if real_a4 else ColorConfig())
            lighting = LightingProcessor(IlluminationConfig(method="combined"), shadow_aware=True) if real_a4 else None

            def build(mode):
                tags = AprilTagLocalizer(layout, scale) if mode != "map" else None
                return LiveLocalization(MapMatcher(campus, search, colors), CameraPreprocessor(preprocess_config),
                                        localization_mode=mode, tag_localizer=tags, lighting=lighting,
                                        map_period_ms=args.map_period_ms)
            return build

        samples = list(synthetic_samples(campus, layout))
        synth_config = replace(config, calibration_path=calibration)
        report = dict(opencv=cv2.__version__, threads=args.threads, map_period_ms=args.map_period_ms,
                      synthetic=run_paired(samples, make_builder(synth_config, .08)))
        if args.donor_frames is not None:
            report["donor_pose_validation"] = evaluate_donor_frames(args.donor_frames, layout)
        if args.video is not None:
            real_config = PhysicalCameraConfig.load(args.camera_config) if args.camera_config else config
            scale = args.print_width_m / campus.width_m if args.print_width_m is not None else None
            builder = make_builder(real_config, scale, real_a4=args.a4_color_profile)
            samples = list(sequential_video(args.video, args.video_frames))
            if not samples:
                raise ValueError("recording contains no frames")
            report["recording"] = dict(path=str(args.video), a4_color_profile=args.a4_color_profile,
                                       results=run_paired(samples, builder))
            if args.scheduled_seconds:
                report["scheduled_replay"] = {mode: scheduled_video(args.video, mode, builder, args.scheduled_seconds)
                                              for mode in MODES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
