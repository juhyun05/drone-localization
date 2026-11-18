from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter, sleep
import csv
import unittest

import numpy as np

from localization.camera_interface import CameraFrame, LatestFrameBuffer
from localization.frame_scheduler import FrameScheduler
from localization.live_localization import LiveResult
from localization.map_matching import Estimate


class FrameSource:
    def __init__(self):
        self.frames = LatestFrameBuffer()
        self.finished = False
        self.image = np.zeros((4, 4, 3), np.uint8)

    def publish(self, frame_id, *, age=0):
        now = perf_counter() - age
        self.frames.publish(CameraFrame(self.image, now, frame_id, "test", now))

    def wait_latest(self, after_id, timeout):
        return self.frames.wait_latest(after_id, timeout)


class FrameSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.source = FrameSource()
        self.events = Queue()
        self.release = Event()
        self.started = Event()

    def scheduler(self, process=None, **options):
        scheduler = FrameScheduler(self.source, process or self.success,
                                   on_event=self.events.put, **options)
        self.addCleanup(scheduler.close)
        self.addCleanup(self.release.set)
        return scheduler

    def success(self, _frame):
        return LiveResult(Estimate(True, "matched"), None, 0)

    def next_event(self):
        return self.events.get(timeout=2)

    def assert_no_event(self, timeout=0.1):
        with self.assertRaises(Empty):
            self.events.get(timeout=timeout)

    def test_buffer_replaces_old_frame_and_supports_independent_cursors(self):
        self.source.publish(1)
        self.source.publish(2)
        frame = self.source.frames.wait_latest(0, 0)
        self.assertEqual(frame.frame_id, 2)
        self.assertIsNone(self.source.frames.wait_latest(2, 0))
        self.assertIs(self.source.frames.wait_latest(1, 0), frame)
        self.assertFalse(frame.image.flags.writeable)

    def test_slow_processing_takes_latest_without_queue_or_repeats(self):
        def process(frame):
            if frame.frame_id == 1:
                self.started.set()
                self.release.wait(2)
            return self.success(frame)

        scheduler = self.scheduler(process)
        self.source.publish(1)
        scheduler.start()
        self.assertTrue(self.started.wait(1))
        for frame_id in range(2, 31):
            self.source.publish(frame_id)
        self.release.set()
        self.assertEqual(self.next_event().frame.frame_id, 1)
        last = self.next_event()
        self.assertEqual(last.frame.frame_id, 30)
        self.assertEqual(last.skipped, 28)
        self.assert_no_event()

    def test_100_and_150_ms_limit_starts_and_select_latest_after_wait(self):
        for period in (100, 150, 200):
            with self.subTest(period=period):
                self.source = FrameSource()
                self.source.publish(1)
                scheduler = self.scheduler(period_ms=period)
                scheduler.start()
                first = self.next_event()
                self.source.publish(2)
                self.source.publish(3)
                second = self.next_event()
                self.assertEqual(second.frame.frame_id, 3)
                self.assertGreaterEqual(second.started_at, first.started_at + period / 1000)
                scheduler.close()

    def test_overrun_proceeds_without_starting_another_full_period(self):
        def process(frame):
            if frame.frame_id == 1:
                self.started.set()
                self.release.wait(2)
            return self.success(frame)

        scheduler = self.scheduler(process, period_ms=150)
        self.source.publish(1)
        scheduler.start()
        self.assertTrue(self.started.wait(1))
        self.assert_no_event(0.18)
        self.source.publish(2)
        self.release.set()
        first, second = self.next_event(), self.next_event()
        self.assertLess(second.started_at - first.ended_at, 0.12)

    def test_stale_frame_rejected_once_and_work_resumes_on_new_input(self):
        self.source.publish(1, age=1)
        scheduler = self.scheduler(max_frame_age_ms=300)
        scheduler.start()
        rejected = self.next_event()
        self.assertEqual(rejected.status, "stale")
        self.assertIsNone(rejected.result)
        self.assertIsNone(rejected.started_at)
        self.assertGreater(rejected.frame_age_ms, 300)
        self.assert_no_event()
        self.source.publish(2)
        self.assertEqual(self.next_event().status, "processed")
        self.assert_no_event()

    def test_report_separates_match_modes_and_counts_losses_once(self):
        from localization.frame_scheduler import PerformanceStats
        from localization.tracking_manager import TrackingResult, TrackingState

        tracked = TrackingResult(0, TrackingState.TRACKING, "global", Estimate(True, "ok"),
                                 True, "ok", None, None, None, None, False, 1, 0, 1)
        lost = replace(tracked, state=TrackingState.RELOCALIZING, attempt="local",
                       measurement=Estimate(False, "lost"), measurement_used=False)
        results = iter((
            LiveResult(tracked.measurement, tracked, 1, match_mode="global", matching_ms=11),
            LiveResult(lost.measurement, lost, 1, match_mode="local", matching_ms=3),
            LiveResult(None, replace(lost, attempt="none"), 1),
        ))
        scheduler = self.scheduler(lambda _: next(results))
        stats = PerformanceStats()
        scheduler.start()
        for index in (1, 2, 3):
            self.source.publish(index)
            stats.add(self.next_event())
        summary = stats.summary(3, 1)
        self.assertEqual(summary["stages_ms"]["global_matching"]["mean"], 11)
        self.assertEqual(summary["stages_ms"]["local_matching"]["mean"], 3)
        self.assertEqual(summary["tracking_loss_count"], 1)
        self.assertEqual(summary["matching_success_rate"], 0.5)
        self.assertEqual(scheduler.latest_attempt[1:], (2, lost.measurement))

    def test_benchmark_csv_marks_results_completed_after_window(self):
        from tools.benchmark_live_pipeline import measure_mode

        class OneFrameSource(FrameSource):
            error = ""
            received = 1

            def start(self):
                self.publish(1)

            def close(self):
                self.finished = True
                self.frames.clear(finished=True)

        class SlowLocalizer:
            def process(self, _frame):
                sleep(0.4)
                return LiveResult(Estimate(True, "matched"), None, 400)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "performance.csv"
            report = measure_mode(OneFrameSource(), SlowLocalizer(), path, period_ms=0,
                                  seconds=0.20, max_frame_age_ms=300)
            with path.open(encoding="utf-8", newline="") as handle:
                row, = csv.DictReader(handle)
        self.assertEqual(row["status"], "drained")
        self.assertEqual(report["processed_frame_count"], 0)
        self.assertEqual(report["skipped_frame_count"], 0)
        self.assertEqual(report["drained_frame_count"], 1)
        self.assertFalse(report["complete"])


if __name__ == "__main__":
    unittest.main()
