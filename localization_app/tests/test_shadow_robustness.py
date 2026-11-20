from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from localization.camera_interface import CameraFrame
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.illumination_correction import IlluminationConfig, IlluminationCorrector
from localization.lighting_pipeline import LightingProcessor
from localization.live_localization import LiveLocalization
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import MapMatcher, SearchConfig, _color_iou, angle_error
from localization.shadow_detection import detect_shadows
from simulation.lighting import LIGHTING_CASES, simulate_lighting


class SyntheticLightingTest(unittest.TestCase):
    def test_shadow_simulation_keeps_input_and_moves_mask(self):
        source = np.full((90, 120, 3), (235, 238, 240), np.uint8)
        for case in LIGHTING_CASES:
            with self.subTest(case=case):
                image, truth = simulate_lighting(source, case)
                self.assertEqual(image.shape, source.shape)
                self.assertEqual(truth.shape, source.shape[:2])
                self.assertEqual(truth.dtype, np.bool_)
                self.assertTrue(np.all(source == (235, 238, 240)))
        first, mask1 = simulate_lighting(source, "moving", phase=0)
        second, mask2 = simulate_lighting(source, "moving", phase=1)
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.array_equal(mask1, mask2))


class WeightedColorIoUTest(unittest.TestCase):
    def test_nonuniform_weights_match_the_direct_color_iou(self):
        area = np.array([[0, 1, 2, 0], [1, 1, 0, 2], [0, 2, 0, 1]], np.uint8)
        labels = np.array([[1, 2, 0], [0, 1, 2]], np.uint8)
        weights = np.array([[0.1, 1, 0.7], [0, 0.5, 0.3]], np.float32)
        visible = np.ones_like(weights)
        layers = {}
        for label in (1, 2):
            plane = (area == label).astype(np.uint8)
            layers[label] = (plane.astype(np.float32), cv2.boundingRect(plane))
        observed = weights[labels > 0].sum()
        scores = _color_iou((area > 0).astype(np.float32), layers, labels, (0, 0, visible),
                            observed, weights=weights)
        for row in range(scores.shape[0]):
            for col in range(scores.shape[1]):
                template = area[row:row + 2, col:col + 3]
                intersection = weights[(template == labels) & (labels > 0)].sum()
                union = weights[template > 0].sum() + observed - intersection
                self.assertAlmostEqual(scores[row, col], intersection / union, places=6)


class ShadowRobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campus = CampusMap.load()
        cls.raster = MapRaster(cls.campus, 320)
        cls.config = CameraConfig(width_px=320, height_px=240)
        cls.matcher = MapMatcher(cls.campus, SearchConfig(auto_scale=True))
        cls.camera = SyntheticCamera(cls.raster, cls.config)
        cls.combined = LightingProcessor(IlluminationConfig("combined"), shadow_aware=True)

    def assert_pose(self, result, pose, *, position=0.04, yaw=9):
        self.assertTrue(result.success, result)
        self.assertLess(math.dist((result.x_m, result.y_m), (pose.x_m, pose.y_m)), position)
        self.assertLess(angle_error(result.yaw_deg, pose.yaw_deg), yaw)

    def test_clear_frame_keeps_existing_result_for_combined_and_all_one_weights(self):
        pose = Pose(1.73, 2.02, 37)
        frame = self.camera.capture(pose)
        baseline = self.matcher.estimate(frame)
        self.assert_pose(baseline, pose)
        self.assertEqual(self.matcher.estimate(frame, confidence_mask=np.ones(frame.shape[:2])), baseline)
        combined = self.combined.apply(frame)
        self.assertEqual(combined.shadow.shadow_ratio, 0)
        self.assertEqual(self.matcher.estimate(combined.image, confidence_mask=combined.confidence), baseline)

    def test_uniform_darkening_recovers_with_constancy_without_detecting_a_local_shadow(self):
        pose = Pose(1.73, 2.02, 37)
        frame, _ = simulate_lighting(self.camera.capture(pose), "uniform")
        self.assertFalse(self.matcher.estimate(frame).success)
        self.assertEqual(detect_shadows(frame).shadow_ratio, 0)
        corrected = IlluminationCorrector(IlluminationConfig("constancy")).apply(frame)
        self.assert_pose(self.matcher.estimate(corrected), pose)

    def test_partial_and_soft_shadows_restore_a_missed_pose(self):
        pose = Pose(1.05, 0.81, 115)
        clean = self.camera.capture(pose)
        for case in ("local", "soft"):
            with self.subTest(case=case):
                frame, truth = simulate_lighting(clean, case)
                self.assertFalse(self.matcher.estimate(frame).success)
                observation = self.combined.apply(frame)
                predicted = observation.shadow.shadow_mask
                self.assertGreater((predicted & truth).sum() / (predicted | truth).sum(), 0.70)
                self.assert_pose(self.matcher.estimate(observation.image, confidence_mask=observation.confidence),
                                 pose)

    def test_dark_building_and_thin_road_are_not_used_as_shadow_evidence(self):
        image = np.full((240, 320, 3), 243, np.uint8)
        cv2.rectangle(image, (110, 70), (200, 180), (24, 45, 72), -1)
        cv2.line(image, (0, 200), (319, 200), (65, 65, 65), 5)
        evidence = detect_shadows(image)
        self.assertLess(evidence.shadow_ratio, 0.01)
        self.assertAlmostEqual(float(evidence.confidence_mask[110, 150]), 1)

    def test_broad_near_black_area_is_downweighted_even_without_usable_paper_pixels(self):
        image = np.full((240, 320, 3), 238, np.uint8)
        image[:, :220] = 7
        evidence = detect_shadows(image)
        self.assertTrue(evidence.shadow_mask[100, 100])
        self.assertLess(evidence.confidence_mask[100, 100], 0.1)
        self.assertLess(evidence.effective_coverage, 0.35)
        pose = Pose(1.73, 2.02, 37)
        occluded = self.camera.capture(pose)
        occluded[70:] = 0
        observation = self.combined.apply(occluded)
        result = MapMatcher(self.campus).estimate(
            observation.image, (pose.x_m, pose.y_m), radius_m=0.13,
            prior_yaw=pose.yaw_deg, yaw_radius_deg=10, confidence_mask=observation.confidence)
        self.assertEqual(result.reason, "insufficient reliable observations")

    def test_rejects_severe_loss_or_tiny_effective_observation(self):
        pose = Pose(1.05, 0.81, 115)
        clean = self.camera.capture(pose)
        severe, _ = simulate_lighting(clean, "severe")
        observation = self.combined.apply(severe)
        estimate = self.matcher.estimate(observation.image, confidence_mask=observation.confidence)
        self.assertFalse(estimate.success, estimate)
        tiny = np.zeros(clean.shape[:2], np.float32)
        tiny[30:60, 10:35] = 1
        result = self.matcher.estimate(clean, confidence_mask=tiny)
        self.assertEqual(result.reason, "insufficient reliable observations")
        with self.assertRaises(ValueError):
            self.matcher.estimate(clean, confidence_mask=np.full(clean.shape[:2], np.nan))

    def test_tracking_handles_moving_shadow_height_change_and_measurement_loss(self):
        pose = Pose(1.73, 2.02, 37)
        localizer = LiveLocalization(self.matcher, CameraPreprocessor(PhysicalCameraConfig.load()),
                                     lighting=self.combined)
        for i, height in enumerate((0.50, 0.53, 0.56, 0.60)):
            camera = SyntheticCamera(self.raster, replace(self.config, height_m=height))
            frame, _ = simulate_lighting(camera.capture(pose), "moving", phase=i / 3)
            result = localizer.process(CameraFrame(frame, i * 0.25, i + 1, "synthetic"))
            self.assertTrue(result.tracking.measurement_used, result.tracking)
            self.assert_pose(result.measurement, pose)
        dark = np.zeros_like(frame)
        for i in range(4, 7):
            missed = localizer.process(CameraFrame(dark, i * 0.25, i + 1, "synthetic"))
            self.assertFalse(missed.tracking.measurement_used)
        self.assertEqual(missed.tracking.state.value, "RELOCALIZING")
        restored = localizer.process(CameraFrame(camera.capture(pose), 1.75, 8, "synthetic"))
        self.assertTrue(restored.tracking.relocalized)
        self.assert_pose(restored.measurement, pose)

    def test_debug_view_is_synchronized_and_recording_keeps_scalar_metrics(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from localization.camera_recording import CameraRecorder
        from visualization.live_camera_viewer import LiveCameraViewer

        processor = LightingProcessor(IlluminationConfig("combined"), shadow_aware=True, debug=True)
        localizer = LiveLocalization(self.matcher, CameraPreprocessor(PhysicalCameraConfig.load()),
                                     global_only=True, lighting=processor)
        pose = Pose(1.73, 2.02, 37)
        frame, _ = simulate_lighting(self.camera.capture(pose), "soft")

        class Video:
            def close(self):
                pass

        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.avi"
            recorder = CameraRecorder(path, {})
            viewer = LiveCameraViewer(Video(), localizer, self.camera)
            try:
                viewer.frame = CameraFrame(frame, 0, 1, "synthetic")
                viewer.result = localizer.process(viewer.frame)
                recorder.write(viewer.frame, viewer.result)
                viewer._render(0, None)
                self.assertEqual(viewer.image.get_array().shape, (480, 640, 3))
                self.assertIn("Weighted IoU", viewer.status.get_text())
                self.assertIn("Shadow ratio", viewer.status.get_text())
            finally:
                viewer.close()
                recorder.close()
                plt.close(viewer.figure)
            with path.with_suffix(".csv").open(encoding="utf-8", newline="") as report:
                entry = json.loads(next(csv.DictReader(report))["result"])
            self.assertIsNone(entry["diagnostic"])
            self.assertGreater(entry["lighting"]["shadow_ratio"], 0)
