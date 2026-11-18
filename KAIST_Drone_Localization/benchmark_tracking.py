"""Evaluate timestamped trajectories independently of the tracker."""

import csv
from dataclasses import replace
import math
from time import perf_counter, sleep

import numpy as np

from localization.camera_simulator import SyntheticCamera
from localization.map_matching import angle_error
from localization.tracking_manager import TrackingManager, TrackingState
from simulation.drone_trajectory import DroneTrajectory, capture_sample
from simulation.playback import latest_due_index


class InjectedOutlier:
    """Inject a high-score false local match after running the real matcher."""

    def __init__(self, matcher):
        self.matcher = matcher
        self.active = False

    def estimate(self, frame, prior_xy=None, **kwargs):
        result = self.matcher.estimate(frame, prior_xy, **kwargs)
        if self.active and prior_xy is not None and result.success:
            return replace(result, x_m=result.x_m - 0.7, score=0.98, margin=0.4)
        return result


def compare_matching(matcher, camera, dt_s):
    """Compare full and narrowed local matching on identical subsequent frames."""
    timings = {"global": [], "local full": [], "local narrowed": []}
    for path in ("line", "square", "circle"):
        samples = DroneTrajectory.preset(path, 0.32).samples(dt_s)
        first_frame = camera.capture(samples[0].pose)
        start = perf_counter()
        initial = matcher.estimate(first_frame)
        timings["global"].append(((perf_counter() - start) * 1000, _match_error(initial, samples[0])))
        if not initial.success:
            continue
        frame = camera.capture(samples[1].pose)
        for name, options in (
            ("local full", {}),
            ("local narrowed", {"radius_m": 0.13, "prior_yaw": initial.yaw_deg,
                                "prior_span_m": initial.span_m, "yaw_radius_deg": 30}),
        ):
            start = perf_counter()
            estimate = matcher.estimate(frame, (initial.x_m, initial.y_m), **options)
            timings[name].append(((perf_counter() - start) * 1000, _match_error(estimate, samples[1])))
    print("Same-frame matching comparison (mean/median/p95 ms)")
    for name, rows in timings.items():
        errors = [error for _, error in rows if error is not None]
        print(f"  {name}: {_times([ms for ms, _ in rows])}, "
              f"accepted {len(errors)}/{len(rows)}, RMSE {_rmse(errors)} m")


def run_tracking_benchmark(raster, matcher, camera_config, dt_s=0.25, speed=0.32,
                           csv_path=None, seed=42):
    paths = {name: DroneTrajectory.preset(name, speed) for name in ("line", "square", "circle")}
    sparse = DroneTrajectory.waypoints([(0.89, 0.77), (1.53, 0.77), (1.86, 1.24)], speed)
    cases = [
        ("line_unknown", paths["line"], camera_config, None),
        ("square", paths["square"], camera_config, None),
        ("circle", paths["circle"], camera_config, None),
        ("noise", paths["line"], replace(camera_config, noise_std=4), None),
        ("lighting", paths["circle"], replace(camera_config, brightness=0.86,
                                             channel_gains=(1.02, 0.99, 1.01)), None),
        ("sparse", sparse, camera_config, None),
        ("blackout", paths["square"], camera_config, None),
        ("outlier", paths["line"], camera_config, None),
        ("line_known", paths["line"], camera_config, (1.67, 2.06)),
        ("wrong_prior", paths["line"], camera_config, (0.31, 0.32)),
    ]
    compare_matching(matcher, SyntheticCamera(raster, camera_config, seed), dt_s)
    rows = []
    for name, trajectory, config, initial_xy in cases:
        camera = SyntheticCamera(raster, config, seed)
        instrumented = InjectedOutlier(matcher)
        tracker = TrackingManager(instrumented, initial_xy=initial_xy)
        scenario = []
        for i, sample in enumerate(trajectory.samples(dt_s)):
            frame, timestamp = capture_sample(camera, sample)
            if name == "blackout" and 2 <= i <= 5:
                frame.fill(230)
            instrumented.active = name == "outlier" and i == 2
            result = tracker.step(frame, timestamp)
            scenario.append(_row(name, sample, result, instrumented.active))
        _summary(name, scenario)
        rows.extend(scenario)
    _summary("ALL", rows)

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    _realtime_replay(paths["square"], dt_s, SyntheticCamera(raster, camera_config, seed), matcher)
    _realtime_replay(paths["square"], dt_s, SyntheticCamera(raster, camera_config, seed), matcher,
                     initial_xy=(1.74, 1.75))


def _match_error(measurement, sample):
    if measurement is None or not measurement.success:
        return None
    return math.dist((measurement.x_m, measurement.y_m), (sample.pose.x_m, sample.pose.y_m))


def _row(name, sample, result, injection_requested):
    truth = (sample.pose.x_m, sample.pose.y_m)
    measured = result.measurement
    return {
        "scenario": name, "timestamp_s": sample.timestamp_s, "truth_x": truth[0], "truth_y": truth[1],
        "state": result.state.value, "attempt": result.attempt, "reason": result.reason,
        "measurement_used": result.measurement_used,
        "measurement_x": measured.x_m if measured else None,
        "measurement_y": measured.y_m if measured else None,
        "measurement_error_m": _match_error(measured, sample),
        "filter_x": result.xy_m[0] if result.xy_m else None,
        "filter_y": result.xy_m[1] if result.xy_m else None,
        "filter_error_m": math.dist(result.xy_m, truth) if result.xy_m else None,
        "yaw_error_deg": angle_error(result.yaw_deg, sample.pose.yaw_deg)
        if result.yaw_deg is not None else None,
        "relocalized": result.relocalized, "sigma_m": result.sigma_m,
        "match_ms": result.match_ms, "filter_ms": result.filter_ms, "total_ms": result.total_ms,
        "outlier_injection_requested": injection_requested,
    }


def _times(values):
    if not values:
        return "n/a"
    return "/".join(f"{value:.1f}" for value in (np.mean(values), np.median(values), np.percentile(values, 95)))


def _rmse(values):
    return f"{np.sqrt(np.mean(np.square(values))):.4f}" if values else "n/a"


def _summary(name, rows):
    measured = [row["measurement_error_m"] for row in rows if row["measurement_used"]]
    errors = [row["filter_error_m"] for row in rows if row["filter_error_m"] is not None]
    yaw = [row["yaw_error_deg"] for row in rows if row["yaw_error_deg"] is not None]
    lost_since = {}
    durations = []
    episodes = 0
    for row in rows:
        scenario, timestamp = row["scenario"], row["timestamp_s"]
        if row["state"] in (TrackingState.PREDICT_ONLY.value, TrackingState.RELOCALIZING.value):
            if scenario not in lost_since:
                episodes += 1
                lost_since[scenario] = timestamp
        elif scenario in lost_since and row["measurement_used"]:
            durations.append(timestamp - lost_since.pop(scenario))
    predicted = sum(row["filter_error_m"] is not None and not row["measurement_used"] for row in rows)
    relocalizations = sum(row["relocalized"] for row in rows)
    global_retries = sum(row["attempt"] == "global" and
                         (row["relocalized"] or row["state"] == TrackingState.RELOCALIZING.value)
                         for row in rows)
    global_ms = [row["match_ms"] for row in rows if row["attempt"] == "global"]
    local_ms = [row["match_ms"] for row in rows if row["attempt"] == "local"]
    tracking_ms = [row["match_ms"] for row in rows if row["attempt"] == "local" and row["measurement_used"]]
    total = [row["total_ms"] for row in rows]
    count = len(rows)
    print(f"{name}: measurement {len(measured)}/{count}, tracking {len(measured)}/{count}, "
          f"prediction {predicted}/{count}, filter coverage {len(errors)}/{count}, "
          f"filter <=5 cm {sum(e <= 0.05 for e in errors)}/{count}, "
          f"bad accepted >5 cm {sum(e > 0.05 for e in measured)}/{len(measured)}")
    print(f"  measurement/filter RMSE {_rmse(measured)}/{_rmse(errors)} m, "
          f"max/p95 {f'{max(errors):.3f}/{np.percentile(errors, 95):.3f}' if errors else 'n/a'} m, "
          f"yaw mean {f'{np.mean(yaw):.2f}' if yaw else 'n/a'} deg ({len(yaw)}/{count})")
    print(f"  recovery {len(durations)}/{episodes}, outage s {', '.join(f'{t:.2f}' for t in durations) or 'n/a'}, "
          f"global reacquisitions {relocalizations}/{global_retries} attempts")
    print(f"  ms mean/median/p95: global {_times(global_ms)}, local {_times(local_ms)}, "
          f"accepted local {_times(tracking_ms)}, filter {_times([row['filter_ms'] for row in rows])}, "
          f"total {_times(total)}; compute {1000 / np.mean(total):.1f} FPS")
    for row in rows:
        if row["outlier_injection_requested"]:
            print(f"  outlier frame t={row['timestamp_s']:.2f}: {row['reason']}, "
                  f"accepted={row['measurement_used']}, measured error={row['measurement_error_m']}")


def _realtime_replay(path, dt_s, camera, matcher, initial_xy=None):
    samples = path.samples(dt_s)
    timestamps = [sample.timestamp_s for sample in samples]
    tracker = TrackingManager(matcher, initial_xy=initial_xy)
    index, processed = -1, 0
    start = perf_counter()
    while index < len(samples) - 1:
        elapsed = perf_counter() - start
        due = latest_due_index(timestamps, elapsed, index)
        if due is None:
            sleep(max(0, timestamps[index + 1] - elapsed))
            continue
        index = due
        tracker.step(*capture_sample(camera, samples[index]))
        processed += 1
    elapsed = perf_counter() - start
    initialization = "known prior" if initial_xy else "cold start"
    print(f"Real-time square replay ({initialization}) at {1 / dt_s:.1f} input FPS: "
          f"processed {processed}/{len(samples)}, "
          f"dropped {len(samples) - processed}, throughput {processed / elapsed:.2f} FPS including waits")
