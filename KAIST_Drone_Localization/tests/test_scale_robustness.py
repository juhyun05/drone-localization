import math
import unittest
from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import MatchDiagnostics, MapMatcher, SearchConfig, angle_error
from localization.tracking_manager import TrackingConfig, TrackingManager


class ScaleRobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campus = CampusMap.load()
        cls.raster = MapRaster(cls.campus, 320)
        cls.camera = CameraConfig()
        cls.pose = Pose(1.73, 2.02, 37)

    def capture(self, ratio, pose=None, **camera_options):
        config = replace(self.camera, height_m=self.camera.height_m * ratio, **camera_options)
        return SyntheticCamera(self.raster, config).capture(pose or self.pose), config.footprint_m

    def check_pose(self, estimate, span, xy_tolerance=0.05, span_tolerance=0.05, pose=None):
        pose = pose or self.pose
        self.assertTrue(estimate.success, estimate)
        self.assertLess(math.dist((estimate.x_m, estimate.y_m), (pose.x_m, pose.y_m)), xy_tolerance)
        self.assertLessEqual(angle_error(estimate.yaw_deg, pose.yaw_deg), 8)
        self.assertLess(abs(estimate.span_m - span) / span, span_tolerance)

    def test_global_recovers_span_outside_fixed_range_and_at_base_height(self):
        automatic = MapMatcher(self.campus, SearchConfig(auto_scale=True))
        fixed = MapMatcher(self.campus)
        for ratio in (0.65, 1.0, 1.65):
            with self.subTest(ratio=ratio):
                frame, span = self.capture(ratio)
                self.check_pose(automatic.estimate(frame), span)
                old = fixed.estimate(frame)
                if ratio == 1:
                    self.check_pose(old, span)
                else:
                    self.assertTrue(not old.success or abs(old.span_m / span - 1) > 0.15)

    def test_gradual_height_changes_stay_local_without_position_drift(self):
        tracker = TrackingManager(MapMatcher(self.campus, SearchConfig(auto_scale=True)))
        for i, ratio in enumerate((1.0, 1.06, 1.12, 1.20, 1.28)):
            frame, span = self.capture(ratio)
            result = tracker.step(frame, i * 0.25)
            self.assertEqual(result.attempt, "local" if i else "global")
            self.assertTrue(result.measurement_used, result)
            self.check_pose(result.measurement, span)
            self.assertLess(math.dist(result.xy_m, (self.pose.x_m, self.pose.y_m)), 0.05)

    def test_relocalization_updates_span_after_sudden_height_change(self):
        tracker = TrackingManager(MapMatcher(self.campus, SearchConfig(auto_scale=True)),
                                  TrackingConfig(max_misses=1))
        frame, _ = self.capture(0.65)
        self.assertTrue(tracker.step(frame, 0).measurement_used)
        lost = tracker.step(np.full_like(frame, 255), 0.3)
        self.assertEqual(lost.state.value, "RELOCALIZING")
        self.assertFalse(lost.measurement_used)
        frame, span = self.capture(1.65)
        recovered = tracker.step(frame, 0.7)
        self.assertEqual(recovered.attempt, "global")
        self.assertTrue(recovered.relocalized)
        self.check_pose(recovered.measurement, span)
        self.assertEqual(recovered.span_m, recovered.measurement.span_m)

    def test_one_miss_expands_scale_range_without_losing_position_prior(self):
        tracker = TrackingManager(MapMatcher(self.campus, SearchConfig(auto_scale=True)))
        frame, _ = self.capture(1.30)
        self.assertTrue(tracker.step(frame, 0).measurement_used)
        frame, span = self.capture(1.75)
        miss = tracker.step(frame, 0.25)
        self.assertEqual(miss.state.value, "PREDICT_ONLY")
        recovered = tracker.step(frame, 0.50)
        self.assertEqual(recovered.attempt, "local")
        self.assertTrue(recovered.measurement_used, recovered)
        self.check_pose(recovered.measurement, span)

    def test_processed_camera_frame_displays_scale_and_rotated_footprint(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from localization.camera_interface import CameraFrame
        from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
        from localization.camera_stream import StreamStatus, XiaoCameraSource
        from localization.live_localization import LiveLocalization
        from visualization.live_camera_viewer import LiveCameraViewer

        config = SearchConfig(auto_scale=True)
        localizer = LiveLocalization(MapMatcher(self.campus, config),
                                     CameraPreprocessor(PhysicalCameraConfig.load()), global_only=True)
        image, span = self.capture(1.55)
        encoded = cv2.imencode(".jpg", image)[1]
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        now = perf_counter()
        frame = CameraFrame(bgr, now, 1, "xiao", now, connection=1)
        viewer = LiveCameraViewer(XiaoCameraSource("http://127.0.0.1/stream"), localizer,
                                  SyntheticCamera(self.raster, self.camera),
                                  nominal_span_m=config.nominal_span_m, physical_per_map=0.18 / 2.5)
        try:
            viewer.frame = viewer.preview_frame = frame
            viewer.result = localizer.process(frame)
            self.check_pose(viewer.result.measurement, span)
            viewer._render(now, StreamStatus("streaming", "", 1, 0, None, 0, 0, 0, 1, 1, now))
            self.assertIn("Scale", viewer.status.get_text())
            self.assertIn("cm on print", viewer.status.get_text())
            points = viewer.map_axes.lines[-1].get_xydata()
            self.assertEqual(len(points), 5)
            self.assertAlmostEqual(math.dist(*points[:2]), viewer.result.measurement.span_m)
            np.testing.assert_allclose(points[0], points[-1])
        finally:
            viewer.close()
            plt.close(viewer.figure)

    def test_rotated_view_with_noise_blur_and_color_shift(self):
        pose = Pose(self.pose.x_m, self.pose.y_m, 112)
        frame, span = self.capture(1.35, pose, noise_std=2, brightness=0.93,
                                   channel_gains=(1.01, 0.995, 1.01))
        frame = cv2.GaussianBlur(frame, (7, 7), 1.8)
        self.check_pose(MapMatcher(self.campus, SearchConfig(auto_scale=True)).estimate(frame),
                        span, span_tolerance=0.07, pose=pose)

    def test_distinct_location_and_scale_with_same_pattern_are_ambiguous(self):
        small_layout = [
            ("#f07a5e", [(-0.13, -0.11), (0.03, -0.11), (0.03, -0.02), (-0.13, -0.02)]),
            ("#58b1ec", [(0.09, 0.02), (0.16, 0.02), (0.16, 0.12), (0.09, 0.12)]),
            ("#74df53", [(-0.13, 0.06), (-0.05, 0.06), (-0.05, 0.12), (-0.13, 0.12)]),
        ]
        buildings = [dict(color=color, polygon_xy_m=[(x + scale * dx, y + scale * dy)
                                                        for dx, dy in polygon])
                     for x, y, scale in ((0.6, 0.8, 1), (1.8, 1.65, 1.5))
                     for color, polygon in small_layout]
        campus = CampusMap(2.5, 2.5, buildings, [], {})
        config = replace(self.camera, height_m=self.camera.height_m / math.sqrt(2))
        frame = SyntheticCamera(MapRaster(campus, 320), config).capture(Pose(0.6, 0.8, 0))
        diagnostics = MatchDiagnostics()
        estimate = MapMatcher(campus, SearchConfig(auto_scale=True)).estimate(
            frame, diagnostics=diagnostics)
        self.assertFalse(estimate.success, estimate)
        self.assertEqual(estimate.reason, "ambiguous pose")
        self.assertGreaterEqual(len(diagnostics.candidates), 2)

    def test_nearby_competitor_beyond_position_separation_is_verified(self):
        rectangles = [
            ("#f07a5e", -0.024, -0.020, -0.006, -0.004),
            ("#58b1ec", 0.008, 0.004, 0.025, 0.022),
            ("#74df53", -0.025, 0.007, -0.014, 0.022),
        ]
        buildings = [dict(color=color, polygon_xy_m=[(x + x0, 0.15 + y0), (x + x1, 0.15 + y0),
                                                       (x + x1, 0.15 + y1), (x + x0, 0.15 + y1)])
                     for x in (0.12, 0.22) for color, x0, y0, x1, y1 in rectangles]
        campus = CampusMap(0.35, 0.30, buildings, [], {})
        span = 0.072
        config = replace(self.camera, height_m=self.camera.height_m * span / self.camera.footprint_m)
        frame = SyntheticCamera(MapRaster(campus, 1200), config).capture(Pose(0.12, 0.15, 0))
        search = SearchConfig(auto_scale=True, pixels_per_m=400, nominal_span_m=span,
                              scale_min=0.85, scale_max=1.18, scale_steps=7)
        matcher = MapMatcher(campus, search)
        estimate = matcher.estimate(frame)
        self.assertFalse(estimate.success, estimate)
        self.assertEqual(estimate.reason, "ambiguous pose")
        diagnostics = MatchDiagnostics()
        local = matcher.estimate(frame, (0.17, 0.15), radius_m=0.10, prior_yaw=0,
                                 yaw_radius_deg=1, prior_span_m=span, diagnostics=diagnostics)
        self.assertEqual(local.reason, "ambiguous pose")
        self.assertAlmostEqual(abs(diagnostics.runner_up.x_m - diagnostics.candidates[0].x_m), 0.10)

    def test_refinement_respects_yaw_search_window(self):
        frame, span = self.capture(1.0, Pose(self.pose.x_m, self.pose.y_m, 27.5))
        matcher = MapMatcher(self.campus, SearchConfig(auto_scale=True))
        estimate = matcher.estimate(frame, (self.pose.x_m, self.pose.y_m), prior_yaw=20,
                                    yaw_radius_deg=5, prior_span_m=span)
        self.assertTrue(estimate.success, estimate)
        self.assertLessEqual(angle_error(estimate.yaw_deg, 20), 5)

    def test_severe_perspective_can_invalidate_uniform_scale_fit(self):
        frame, _ = self.capture(1.25)
        corners = np.float32([[0, 0], [239, 0], [239, 179], [0, 179]])
        tilted = np.float32([[92, 6], [147, 6], [239, 179], [0, 179]])
        transform = cv2.getPerspectiveTransform(corners, tilted)
        warped = cv2.warpPerspective(frame, transform, (240, 180), borderValue=(237, 241, 233))
        automatic = MapMatcher(self.campus, SearchConfig(auto_scale=True))
        self.assertTrue(automatic.estimate(frame).success)
        result = automatic.estimate(warped)
        self.assertFalse(result.success, result)
        self.assertEqual(result.reason, "low match score")

    def test_auto_scale_configuration_must_have_an_ordered_range(self):
        with self.assertRaises(ValueError):
            SearchConfig(auto_scale=True, scale_min=2.0, scale_max=1.0)

    def test_insufficient_observation_is_rejected(self):
        matcher = MapMatcher(self.campus, SearchConfig(auto_scale=True))
        image = np.full((180, 240, 3), 255, np.uint8)
        image[5:10, 5:10] = self.campus.palette[0]
        result = matcher.estimate(image)
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "insufficient visible buildings")
