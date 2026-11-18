"""Measure a camera stream from the PC, with optional global matching."""

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter, process_time

import numpy as np
import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization.camera_stream import XiaoCameraSource


def measure(source, seconds, localizer=None, *, preview=False, csv_path=None):
    intervals, decode, match = [], [], []
    last = None
    source.start()
    start, cpu = perf_counter(), process_time()
    try:
        while perf_counter() - start < seconds:
            frame = source.read(timeout=min(0.2, max(0, seconds - (perf_counter() - start))))
            if preview:
                if frame is not None:
                    cv2.imshow("XIAO camera (Esc to exit)", frame.image)
                elif (last is not None and (source.status().state != "streaming"
                                            or perf_counter() - last.received_at >= 2)):
                    cv2.imshow("XIAO camera (Esc to exit)", np.zeros_like(last.image))
                if cv2.waitKey(1) & 0xff == 27:
                    break
            if frame is None:
                if source.status().state == "failed":
                    break
                continue
            decode.append(frame.decode_ms)
            if last is not None and last.connection == frame.connection and frame.frame_id == last.frame_id + 1:
                intervals.append((frame.timestamp - last.timestamp) * 1000)
            last = frame
            if localizer:
                match.append(localizer.process(frame).processing_ms)
        stats = source.status()
        elapsed = perf_counter() - start
        cpu_percent = (process_time() - cpu) / elapsed * 100
        print(f"{stats.received} decoded in {elapsed:.2f} s, average {stats.received / elapsed:.2f} FPS; "
              f"latest-window {stats.receive_fps:.2f} FPS, process CPU {cpu_percent:.1f}%")
        print(f"state={stats.state}, overwrites={stats.overwritten}, sender gaps={stats.source_gaps}, "
              f"bad JPEG={stats.decode_errors}, disconnects={stats.disconnects}, "
              f"reconnections={stats.reconnections}, error={stats.error or '-'}")
        for name, values in (("Observed consecutive receive interval", intervals),
                             ("Consumed JPEG decode", decode), ("Preprocessing + global matching", match)):
            if values:
                print(f"{name}: mean {np.mean(values):.2f} ms, p95 {np.percentile(values, 95):.2f} ms")
        age = perf_counter() - stats.last_received_at if stats.last_received_at is not None else None
        if last is not None:
            height, width = last.image.shape[:2]
            print(f"Last consumed frame: {width} x {height}, {last.image.dtype} BGR, "
                  f"PC receive age {perf_counter() - last.received_at:.3f} s")
        if csv_path:
            from datetime import datetime

            row = dict(measured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                       requested_s=seconds, elapsed_s=elapsed, received=stats.received,
                       receive_fps=stats.received / elapsed, consumed=len(decode),
                       state=stats.state, error=stats.error, buffer_overwrites=stats.overwritten,
                       sender_gaps=stats.source_gaps, decode_errors=stats.decode_errors,
                       disconnects=stats.disconnects, reconnections=stats.reconnections,
                       latest_receive_age_s=age, cpu_percent=cpu_percent,
                       width=last.image.shape[1] if last is not None else None,
                       height=last.image.shape[0] if last is not None else None,
                       localization_fps=len(match) / elapsed if localizer else None)
            for name, values in (("receive_interval", intervals), ("decode", decode), ("match", match)):
                row[f"{name}_mean_ms"] = float(np.mean(values)) if values else None
                row[f"{name}_p95_ms"] = float(np.percentile(values, 95)) if values else None
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("x", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
        return 0 if stats.received else 1
    finally:
        source.close()
        if preview:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--match", action="store_true")
    parser.add_argument("--preview", action="store_true", help="show received frames in an OpenCV window")
    parser.add_argument("--csv", type=Path, help="save metrics to a new CSV file")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    localizer = None
    if args.match:
        from dataclasses import replace
        from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
        from localization.live_localization import LiveLocalization
        from localization.map_loader import CampusMap
        from localization.map_matching import MapMatcher, SearchConfig

        config = PhysicalCameraConfig.load()
        search = SearchConfig()
        if config.nominal_span_m is not None:
            search = replace(search, nominal_span_m=config.nominal_span_m)
        else:
            print("Using synthetic span as an uncalibrated trial.")
        localizer = LiveLocalization(MapMatcher(CampusMap.load(), search),
                                    CameraPreprocessor(config), global_only=True)
    try:
        return measure(XiaoCameraSource(args.url), args.seconds, localizer, preview=args.preview,
                       csv_path=args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
