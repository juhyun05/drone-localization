"""Compare scheduling modes through the same worker used by the live viewer."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
from time import perf_counter, process_time

import numpy as np

from localization.frame_scheduler import FrameScheduler, PerformanceLog, PerformanceStats
from tools.benchmark_environment import benchmark_environment, process_resources


PERIODS = (0, 100, 150, 200)


def receive_cursor(source):
    if hasattr(source, "receive_cursor"):
        return source.receive_cursor()
    return perf_counter(), source.status().received if hasattr(source, "status") else source.received


def measure_mode(source, localizer, csv_path, *, period_ms, seconds, max_frame_age_ms,
                 ground_truth=None, truth_yaw=None, warmup_s=0):
    stats = PerformanceStats(ground_truth=ground_truth, truth_yaw=truth_yaw)
    log = scheduler = None
    start = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_samples = []
    try:
        log = PerformanceLog(csv_path)
        source.start()
        while not source.finished and perf_counter() - start < 30:
            first = source.wait_latest(0, timeout=0.1)
            if first is not None:
                break
        else:
            first = source.peek()
        if first is None:
            status = source.status() if hasattr(source, "status") else None
            error = status.error if status is not None else source.error
            return dict(period_ms=period_ms, requested_s=seconds, complete=False,
                        received_frame_count=0, error=error or "no decoded frames before startup timeout")

        def record(event):
            if event.ended_at is not None and event.ended_at >= deadline:
                event = replace(event, status="drained")
            stats.add(event)
            log_started = perf_counter()
            log.write(event)
            if event.status != "drained":
                log_samples.append((perf_counter() - log_started) * 1000)

        received_before = first.frame_id - 1
        if warmup_s:
            warmup = FrameScheduler(
                source, localizer.process, period_ms=period_ms, max_frame_age_ms=max_frame_age_ms,
                after_id=received_before, deadline=perf_counter() + warmup_s,
            )
            warmup.start()
            try:
                warmup.done.wait()
            finally:
                warmup.close()
            if warmup.error:
                return dict(period_ms=period_ms, requested_s=seconds, complete=False,
                            received_frame_count=0, error=f"warmup failed: {warmup.error}")
            stats.continue_after(warmup.latest_processed)
        resources_start = process_resources()
        gc_start = sum(generation["collections"] for generation in gc.get_stats())
        cpu_start = process_time()
        start, arrived = receive_cursor(source)
        if warmup_s:
            received_before = arrived
        deadline = start + seconds
        scheduler = FrameScheduler(
            source, localizer.process, period_ms=period_ms, max_frame_age_ms=max_frame_age_ms,
            after_id=received_before, deadline=deadline, on_event=record,
        )
        scheduler.start()
        while perf_counter() < deadline and not source.finished:
            if scheduler.done.wait(timeout=min(0.05, max(0, deadline - perf_counter()))):
                break
        end, received = receive_cursor(source)
        elapsed = max(end - start, 1e-9)
        cpu_percent = (process_time() - cpu_start) / elapsed * 100
        scheduler.stop.set()
        resources_end = process_resources()
        gc_collections = sum(generation["collections"] for generation in gc.get_stats()) - gc_start
        input_status = source.status() if hasattr(source, "status") else None
        source_error = source.error if input_status is None else (
            input_status.error if input_status.state == "failed" else "")
    finally:
        if scheduler is not None:
            scheduler.stop.set()
        try:
            source.close()
        finally:
            if scheduler is not None:
                scheduler.close()
            if log is not None:
                log.close()

    report = stats.summary(received - received_before, elapsed)
    error = scheduler.error or source_error
    complete = elapsed >= seconds and not error and report["processed_frame_count"] > 0
    if not error and not complete:
        error = ("input ended before the measurement window" if elapsed < seconds
                 else "no localization completed in the measurement window")
    stream_status = asdict(input_status) if input_status is not None else None
    if stream_status is not None:
        stream_status.pop("overwritten")
    report.update(period_ms=period_ms, requested_s=seconds, complete=complete,
                  process_cpu_percent=cpu_percent, error=error or None,
                  started_utc=started_utc, warmup_s=warmup_s, gc_collections=gc_collections,
                  resources_start=resources_start, resources_end=resources_end,
                  log_write_ms=(dict(mean=float(np.mean(log_samples)),
                                     p95=float(np.percentile(log_samples, 95)))
                                if log_samples else None),
                  stream_status=stream_status,
                  timestamps_preserved=getattr(source, "timestamps_preserved", None))
    return report


def trial_orders(periods, repeats, seed):
    order = list(periods)
    if repeats > 1 or seed is not None:
        random.Random(42 if seed is None else seed).shuffle(order)
    for index in range(repeats):
        shift = index % len(order)
        yield order[shift:] + order[:shift]


def compare_modes(build_source, build_localizer, output, *, seconds, max_frame_age_ms, scenario,
                  input_kind, input_label, ground_truth=None, truth_yaw=None, settings=None,
                  repeats=1, warmup_s=0, seed=None, periods=PERIODS):
    output = Path(output)
    orders = list(trial_orders(periods, repeats, seed))
    trials = []
    for repeat, order in enumerate(orders, 1):
        folder = output / f"repeat_{repeat:02}" if repeats > 1 else output
        for index, period in enumerate(order, 1):
            name = f"performance_{period}ms.csv" if period else "performance_unlimited.csv"
            trials.append((repeat, index, period, folder / name))
    summary_path = output / "performance_summary.json"
    report_path = output / "performance_report.md"
    for path in (*(trial[3] for trial in trials), summary_path, report_path):
        if path.exists():
            raise ValueError(f"benchmark output already exists: {path}; choose a fresh --benchmark-output")
    environment = benchmark_environment(input_kind, input_label)
    results = []
    summary = dict(input_kind=input_kind, input=input_label, scenario=scenario,
                   requested_seconds_per_mode=seconds, max_frame_age_ms=max_frame_age_ms,
                   ground_truth_xy=ground_truth, truth_yaw_deg=truth_yaw,
                   settings=settings, environment=environment, warmup_s=warmup_s,
                   repeat_count=repeats, order_seed=seed if seed is not None else (42 if repeats > 1 else None),
                   orders=orders, modes=results, complete=False)

    def save_summary():
        pending = summary_path.with_suffix(".tmp")
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        pending.replace(summary_path)

    for repeat, index, period, path in trials:
        localizer = build_localizer()
        result = measure_mode(build_source(), localizer, path, period_ms=period, seconds=seconds,
                              max_frame_age_ms=max_frame_age_ms, ground_truth=ground_truth,
                              truth_yaw=truth_yaw, warmup_s=warmup_s)
        result.update(repeat=repeat, execution_order=index, csv=path.relative_to(output).as_posix())
        results.append(result)
        save_summary()
        prefix = f"Run {repeat}/{repeats} | {period or 'Unlimited'}{' ms' if period else ''}"
        if result["complete"]:
            print(f"{prefix} | {result['camera_receive_fps']:.2f} input FPS | "
                  f"{result['localization_fps']:.2f} localization FPS", flush=True)
        else:
            print(f"{prefix} | incomplete | {result['error']}", flush=True)
            if not result["received_frame_count"]:
                break
    summary["complete"] = len(results) == len(trials) and all(result["complete"] for result in results)
    save_summary()
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(_report(summary))
    if not summary["complete"]:
        raise ValueError(f"benchmark incomplete; see {report_path}")


def _report(summary):
    def number(value, factor=1):
        return "—" if value is None else f"{value * factor:.2f}"

    def pair(data):
        return "—" if data is None else f"{data['mean']:.2f} / {data['p95']:.2f}"

    lines = [
        "# Live pipeline timing", "",
        f"Input `{summary['input_kind']}` · scenario `{summary['scenario']}` · "
        f"{summary['requested_seconds_per_mode']:g} s per mode · {summary['warmup_s']:g} s warmup", "",
        "| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | "
        "Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | "
        "Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for result in summary["modes"]:
        stages = result.get("stages_ms", {})
        period = f"{result['period_ms']} ms" if result["period_ms"] else "Unlimited"
        if summary["repeat_count"] > 1:
            period = f"{result['repeat']}. {period}"
        fields = (period, number(result.get("camera_receive_fps")),
                  number(result.get("localization_fps")), pair(result.get("processing_ms")),
                  pair(result.get("frame_age_ms")), pair(stages.get("global_matching")),
                  pair(stages.get("local_matching")), number(result.get("skipped_frame_ratio"), 100),
                  str(result.get("stale_frame_count", "—")),
                  number(result.get("matching_success_rate"), 100),
                  number(result.get("accepted_observation_rate"), 100),
                  str(result.get("tracking_loss_count", "—")),
                  number(result.get("process_cpu_percent")), number(result.get("position_rmse_m")))
        lines.append("| " + " | ".join(fields) + " |")
    failures = [result for result in summary["modes"] if not result["complete"]]
    if failures:
        lines.extend(("", "## Incomplete measurements", ""))
        lines.extend(f"- {row['period_ms']} ms · {row['error']}" for row in failures)
    lines.extend((
        "", "Age starts at PC JPEG receipt or actual video delivery to the shared slot. "
        "Replay delivery lag is measured against the recorded timeline. CPU is process CPU "
        "time divided by elapsed time and can exceed 100%. Results completed after the window "
        "are marked as drained in CSV. Without warmup the first received frame is included. "
        "Headless runs have no GUI render samples.", "",
        "Match rate uses global/local attempts. Accepted rate uses all processed frames. "
        "Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements "
        "with a known fixed reference. The wrong-acceptance rate uses the same reference and "
        "flags position error over 5 cm or yaw error over 15 degrees when available. "
        "Relocalization time uses the observation timeline for recovered episodes. "
        "RSS and CPU resources are recorded in JSON.", "",
    ))
    return "\n".join(lines)
