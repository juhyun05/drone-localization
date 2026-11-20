from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from localization.camera_interface import (
    CameraFrame, LatestFrameBuffer, SyntheticCameraSource, TimedVideoSource, VideoFileSource,
)
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.camera_recording import CameraRecorder
from localization.camera_simulator import CameraConfig, SyntheticCamera
from localization.live_localization import LiveLocalization
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import MapMatcher
from localization.paths import DATA_DIR
from simulation.drone_trajectory import DroneTrajectory
from tools.calibrate_camera import ground_span


class CameraInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.config = PhysicalCameraConfig.load()

    def test_normalizes_grayscale_and_preserves_aspect(self):
        frame = np.zeros((480, 640), np.uint8)
        self.assertEqual(CameraPreprocessor(self.config).apply(frame).shape, (240, 320, 3))
        portrait = np.zeros((640, 480, 4), np.uint8)
        self.assertEqual(CameraPreprocessor(self.config).apply(portrait).shape, (240, 180, 3))
        with self.assertRaises(ValueError):
            CameraPreprocessor(self.config).apply(np.zeros((1, 1, 3), np.float32))

    def test_color_gains_match_float_reference_for_every_pixel_value(self):
        levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = np.stack([levels, levels.T, 255 - levels], axis=-1)
        for brightness, gains in ((1.0, (1.0, 1.0, 1.0)), (1.5, (0.5, 1.0, 2.0)),
                                  (0.83, (1.12, 0.93, 1.35))):
            config = replace(self.config, size=(16, 16), brightness=brightness, gains=gains)
            factors = brightness * np.asarray(gains, dtype=np.float32)
            expected = np.clip(np.rint(image.astype(np.float32) * factors), 0, 255).astype(np.uint8)
            with self.subTest(brightness=brightness, gains=gains):
                np.testing.assert_array_equal(CameraPreprocessor(config).apply(image), expected)

    def test_record_and_replay_uses_original_intervals(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "camera.avi"
            recorder = CameraRecorder(path, {"example": True})
            try:
                for i, timestamp in enumerate((10.0, 10.125, 10.375), start=1):
                    recorder.write(CameraFrame(np.full((32, 48, 3), 90, np.uint8),
                                               timestamp, i, "xiao"))
            finally:
                recorder.close()
            source = VideoFileSource(path)
            try:
                frames = [source.read() for _ in range(3)]
                self.assertTrue(all(isinstance(frame, CameraFrame) for frame in frames))
                self.assertEqual([frame.timestamp for frame in frames], [0.0, 0.125, 0.375])
                self.assertEqual(frames[0].image.shape, (32, 48, 3))
                self.assertIsNone(source.read())
            finally:
                source.close()
            replay = TimedVideoSource(path)
            try:
                replay.start()
                first = replay.wait_latest(0, 1)
                second = replay.wait_latest(first.frame_id, 1)
                third = replay.wait_latest(second.frame_id, 1)
                self.assertTrue(replay.timestamps_preserved)
                self.assertEqual([frame.timestamp for frame in (first, second, third)],
                                 [0.0, 0.125, 0.375])
                self.assertGreaterEqual(third.received_at - first.received_at, 0.33)
                self.assertIsNone(replay.wait_latest(third.frame_id, 1))
            finally:
                replay.close()

    def test_record_flag_accepts_a_file_or_uses_the_shared_recordings_directory(self):
        from main import main

        options = ["--camera-test", "--camera-url", "http://127.0.0.1/stream", "--record"]
        with patch("main.run_physical_camera") as run:
            main(options)
            default_path = run.call_args.args[0].record
            self.assertEqual(default_path.parent, DATA_DIR / "recordings")
            self.assertEqual(default_path.suffix, ".avi")
            main([*options, "my_session.avi"])
            self.assertEqual(run.call_args.args[0].record, Path("my_session.avi"))

    def test_synthetic_frame_reaches_existing_matcher_and_tracker(self):
        campus = CampusMap.load()
        raster = MapRaster(campus, 320)
        trajectory = DroneTrajectory.waypoints([(1.73, 2.02), (1.76, 2.00)], 0.10)
        source = SyntheticCameraSource(SyntheticCamera(raster, CameraConfig()), trajectory.samples(0.1))
        frame = source.read()
        self.assertIsInstance(frame, CameraFrame)
        self.assertGreater(source.read().timestamp, frame.timestamp)
        matcher = MapMatcher(campus)
        processor = CameraPreprocessor(self.config)
        global_result = LiveLocalization(matcher, processor, global_only=True).process(frame)
        self.assertTrue(global_result.measurement.success, global_result)
        self.assertLess(abs(global_result.measurement.x_m - 1.73), 0.05)
        tracker = LiveLocalization(matcher, processor)
        first = tracker.process(frame)
        self.assertTrue(first.tracking.measurement_used, first)
        self.assertEqual(first.match_mode, "global")
        self.assertGreater(first.matching_ms, 0)
        received = perf_counter() - 0.03
        second = tracker.process(CameraFrame(frame.image, 0.26, 2, "xiao", received))
        self.assertEqual(second.match_mode, "local")
        self.assertGreater(second.matching_ms, 0)
        self.assertGreater(second.hsv_ms, 0)
        self.assertGreater(second.localization_ms, second.matching_ms + second.hsv_ms)
        self.assertEqual(tracker.tracker.timestamp, 0.26)
        self.assertEqual(tracker.tracker.filter.position, second.tracking.xy_m)
        dt = second.projected_timestamp - received
        self.assertGreater(dt, 0)
        np.testing.assert_allclose(
            second.projected_xy_m,
            np.asarray(second.tracking.xy_m) + dt * np.asarray(second.tracking.velocity_mps),
        )
        later = tracker.process(CameraFrame(frame.image, 2.5, 2, "xiao", connection=2)).tracking
        self.assertEqual(later.attempt, "global")
        self.assertTrue(later.measurement_used)
        source.close()

    def test_ground_span_uses_known_length(self):
        self.assertAlmostEqual(ground_span(320, 0.25, 20, 120), 0.8)
        with self.assertRaises(ValueError):
            ground_span(320, 0.25, 20, 20)

    def test_tracker_resets_for_input_outage_but_not_skipped_observations(self):
        from localization.map_matching import Estimate

        class Matcher:
            def estimate(self, _frame, prior_xy=None, **_options):
                return Estimate(True, "matched", x_m=1.73, y_m=2.02, yaw_deg=37)

        frames = LatestFrameBuffer()
        image = np.zeros((240, 320, 3), np.uint8)
        tracker = LiveLocalization(Matcher(), CameraPreprocessor(self.config))
        for frame_id, time_s in enumerate((0.0, 0.8, 1.5, 2.2, 4.5, 4.6), 1):
            frames.publish(CameraFrame(image, time_s, frame_id, "test"))
            if frame_id in (1, 4, 6):
                result = tracker.process(frames.peek())
                self.assertEqual(result.match_mode, "local" if frame_id == 4 else "global")

    def test_gap_does_not_reuse_prior_after_scene_change_or_long_outage(self):
        from localization.map_matching import Estimate

        class Matcher:
            def estimate(self, _frame, **options):
                self.last_hint = options.get("recovery_hint")
                return Estimate(True, "matched", x_m=1.73, y_m=2.02, yaw_deg=37, span_m=0.6)

        matcher = Matcher()
        localizer = LiveLocalization(matcher, CameraPreprocessor(self.config))
        first = np.full((240, 320, 3), 120, np.uint8)
        changed = np.full_like(first, 170)
        localizer.process(CameraFrame(first, 0, 1, "test", connection=1, input_epoch=0))
        result = localizer.process(CameraFrame(changed, 2.4, 2, "test", connection=1, input_epoch=1))
        self.assertIsNone(matcher.last_hint)
        self.assertEqual(result.match_mode, "global")
        result = localizer.process(CameraFrame(changed, 6.0, 3, "test", connection=1, input_epoch=2))
        self.assertIsNone(matcher.last_hint)
        self.assertEqual(result.match_mode, "global")

    def test_live_display_renders_korean_socket_error_with_system_font(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        import matplotlib.pyplot as plt
        from localization.camera_stream import StreamStatus, XiaoCameraSource
        from visualization.live_camera_viewer import LiveCameraViewer

        if "Malgun Gothic" not in font_manager.get_font_names():
            self.skipTest("requires the Windows Malgun Gothic font")
        error = "[WinError 10051] 연결할 수 없는 네트워크에서 소켓 작업을 시도했습니다."
        stats = StreamStatus("failed", error, 0, 0, None, 0, 0, 0, 0, 0, None)
        source = XiaoCameraSource("http://127.0.0.1/stream")
        with matplotlib.rc_context({"font.family": ["DejaVu Sans"]}):
            viewer = LiveCameraViewer(source)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    viewer._render(perf_counter(), stats)
                    viewer.figure.canvas.draw()
                self.assertIn(error, viewer.status.get_text())
                missing_glyphs = [str(item.message) for item in caught
                                  if "missing from font" in str(item.message)]
                self.assertEqual(missing_glyphs, [])
            finally:
                viewer.close()
                plt.close(viewer.figure)

    def test_live_display_shows_latest_frame_and_clears_old_result(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from localization.apriltag_detector import TagDetection
        from localization.camera_stream import StreamStatus, XiaoCameraSource
        from localization.live_localization import LiveResult
        from localization.map_matching import Estimate
        from visualization.live_camera_viewer import LiveCameraViewer

        now = perf_counter()
        image = np.zeros((16, 24, 3), np.uint8)
        campus = CampusMap.load()
        camera = SyntheticCamera(MapRaster(campus, 80), CameraConfig())
        source = XiaoCameraSource("http://127.0.0.1/stream")
        viewer = LiveCameraViewer(source, object(), camera)
        try:
            viewer.frame = CameraFrame(image, now - 3, 1, "xiao", now - 3, connection=1)
            tag = TagDetection(3, "tag36h11", ((5, 4), (18, 4), (18, 12), (5, 12)), 1, now - 3)
            viewer.result = LiveResult(Estimate(True, "matched", x_m=1, y_m=1, yaw_deg=0), None, 3000,
                                       tag_detections=(tag,), localization_mode="fusion")
            viewer.preview_frame = CameraFrame(image, now, 2, "xiao", now, connection=1)
            stats = StreamStatus("streaming", "", 2, 0, None, 0, 0, 0, 1, 10, now)
            viewer._render(now, stats)
            self.assertIn("Frame #2", viewer.status.get_text())
            self.assertIn("Localization frame #1", viewer.status.get_text())
            self.assertIs(viewer.displayed_frame, viewer.preview_frame)
            np.testing.assert_array_equal(viewer.image.get_array(), image)
            self.assertIs(viewer.displayed_pose_frame, viewer.frame)
            viewer._render(now + 3, stats)
            self.assertIsNone(viewer.displayed_frame)
            self.assertIsNone(viewer.displayed_pose_frame)
            self.assertIn("No current localization result", viewer.status.get_text())
        finally:
            viewer.close()
            plt.close(viewer.figure)

    def test_live_preview_adds_tag_overlay_when_matching_result_arrives(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from localization.apriltag_detector import TagDetection
        from localization.camera_stream import StreamStatus, XiaoCameraSource
        from localization.live_localization import LiveResult
        from localization.map_matching import Estimate
        from visualization.live_camera_viewer import LiveCameraViewer

        source = XiaoCameraSource("http://127.0.0.1/stream")
        camera = SyntheticCamera(MapRaster(CampusMap.load(), 80), CameraConfig())
        viewer = LiveCameraViewer(source, object(), camera)
        try:
            now = perf_counter()
            image = np.full((120, 160, 3), 50, np.uint8)
            current = CameraFrame(image, 1., 1, "xiao", now, connection=1)
            viewer.preview_frame = current
            stats = StreamStatus("streaming", "", 1, 0, None, 0, 0, 0, 1, 25, now)
            viewer._render(now, stats)
            np.testing.assert_array_equal(viewer.image.get_array(), image)

            viewer.frame = current
            tag = TagDetection(3, "tag36h11", ((50, 40), (90, 40), (90, 80), (50, 80)), 1, 1.)
            viewer.result = LiveResult(Estimate(True, "matched", x_m=1, y_m=1, yaw_deg=0), None, 30,
                                       tag_detections=(tag,), localization_mode="fusion")
            viewer._render(now, stats)
            self.assertIs(viewer.displayed_frame, current)
            self.assertFalse(np.array_equal(viewer.image.get_array(), image))

            viewer.preview_frame = replace(current, frame_id=2, timestamp=2.)
            viewer._render(now, stats)
            self.assertIs(viewer.displayed_frame, viewer.preview_frame)
            np.testing.assert_array_equal(viewer.image.get_array(), image)
        finally:
            viewer.close()
            plt.close(viewer.figure)

    def test_live_preview_repaints_without_redrawing_the_static_figure(self):
        from unittest.mock import patch

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from localization.camera_stream import StreamStatus, XiaoCameraSource
        from visualization.live_camera_viewer import LiveCameraViewer

        source = XiaoCameraSource("http://127.0.0.1/stream")
        viewer = LiveCameraViewer(source)
        try:
            canvas = viewer.figure.canvas
            canvas.draw()
            dock = viewer.image.axes.bbox
            center_x, center_y = int(dock.x0 + dock.width / 2), int(dock.y0 + dock.height / 2)
            image = np.full((240, 320, 3), (255, 0, 0), np.uint8)
            now = perf_counter()
            viewer.preview_frame = CameraFrame(image, now, 1, "xiao", now, connection=1)
            stats = StreamStatus("streaming", "", 1, 0, None, 0, 0, 0, 1, 25, now)
            with patch.object(canvas, "draw", wraps=canvas.draw) as redraw:
                viewer._render(now, stats)
                self.assertEqual(redraw.call_count, 0)
                pixels = np.asarray(canvas.buffer_rgba())
                np.testing.assert_array_equal(pixels[pixels.shape[0] - 1 - center_y, center_x, :3],
                                              (0, 0, 255))
                self.assertIn("Frame #1", viewer.status.get_text())

                smaller = image[:120, :160]
                viewer.preview_frame = CameraFrame(smaller, now, 2, "xiao", now, connection=1)
                viewer._render(now, stats)
                self.assertEqual(redraw.call_count, 1)
        finally:
            viewer.close()
            plt.close(viewer.figure)
