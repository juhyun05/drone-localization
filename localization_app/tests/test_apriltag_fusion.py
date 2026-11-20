import math
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from localization.apriltag_localization import AprilTagLocalizer, DEFAULT_TAG_MAP, TagMap
from localization.apriltag_detector import TagDetection, make_tag_detector, detect_tags
from localization.camera_interface import CameraFrame
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.live_localization import LiveLocalization
from localization.localization_fusion import LocalizationObservation, from_map, select_observation
from localization.map_loader import CampusMap, DEFAULT_MAP
from localization.map_matching import Estimate
from localization.tracking_manager import TrackingConfig, TrackingManager, TrackingState


class NoMap:
    def estimate(self, *args, **kwargs):
        raise AssertionError("map search was disabled")


class AprilTagFusionTest(unittest.TestCase):
    def test_registered_map_and_detector(self):
        layout = TagMap.load(CampusMap.load(DEFAULT_MAP))
        self.assertEqual(set(layout.tags), set(range(39)))
        self.assertEqual(layout.family, "tag36h11")
        detector = make_tag_detector(layout.family)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image = np.full((240, 320), 255, np.uint8)
        image[70:150, 120:200] = cv2.rotate(cv2.aruco.generateImageMarker(dictionary, 7, 80),
                                           cv2.ROTATE_90_CLOCKWISE)
        tags = detect_tags(image, detector, layout.family, 4, 1.5)
        self.assertEqual(len(tags), 1)
        self.assertEqual((tags[0].tag_id, tags[0].frame_id, tags[0].timestamp), (7, 4, 1.5))
        np.testing.assert_allclose(tags[0].corners_px[0], (199, 70), atol=1)

    def test_tilted_single_tag_and_physical_print_scale(self):
        layout = TagMap.load(CampusMap.load())
        solver = AprilTagLocalizer(layout, physical_per_map=0.08)
        tag = layout.tags[0]
        center = np.r_[np.mean(tag.corners, axis=0) + (0.028, -0.02), 0.28]
        rotation = cv2.Rodrigues(np.array([0.17, 0.10, -0.24]))[0] @ np.diag([1., -1., -1.])
        rvec = cv2.Rodrigues(rotation)[0]
        matrix = np.array([[245., 0, 156.], [0, 247., 117.], [0, 0, 1.]])
        distortion = np.array([-0.08, 0.03, 0.001, 0, 0])
        corners = np.c_[tag.corners * 0.08, np.zeros(4)]
        pixels, _ = cv2.projectPoints(corners, rvec, -rotation @ (center * 0.08), matrix, distortion)
        detection = TagDetection(0, layout.family, tuple(map(tuple, pixels.reshape(4, 2))), 1, 0.0)
        pose = solver.solve([detection], (320, 240), (matrix, distortion))
        self.assertTrue(pose.accepted, pose.rejection_reason)
        np.testing.assert_allclose((pose.x_m, pose.y_m), center[:2], atol=1e-4)
        self.assertAlmostEqual(pose.quality_metrics["height_physical_m"], center[2] * 0.08, 5)
        up = rotation.T @ (0., -1., 0.)
        expected = math.degrees(math.atan2(-up[0], up[1])) % 360
        self.assertAlmostEqual(pose.yaw_deg, expected, 3)
        ray = rotation.T @ np.linalg.solve(matrix, (159.5, 119.5, 1.))
        np.testing.assert_allclose(pose.quality_metrics["image_center_offset_m"],
                                   -center[2] * ray[:2] / ray[2], atol=1e-5)
        axes = ((0, -1), (1, 0))
        rotated = solver.solve([detection], (320, 240), (matrix, distortion), axes)
        up = rotation.T @ (-1., 0., 0.)
        expected = math.degrees(math.atan2(-up[0], up[1])) % 360
        self.assertAlmostEqual(rotated.yaw_deg, expected, 3)

    def test_uncalibrated_planar_tag_scores_registered_corners(self):
        layout = TagMap.load(CampusMap.load())
        solver = AprilTagLocalizer(layout, physical_per_map=1.)
        center = layout.tags[6].corners.mean(axis=0) + (.04, .02)
        radians = math.radians(37)
        east = np.array((math.cos(radians), math.sin(radians)))
        north = np.array((-math.sin(radians), math.cos(radians)))
        frame_center = (159.5, 119.5)
        offset = layout.tags[6].corners - center
        pixels = np.column_stack((offset @ east, -offset @ north)) * 800 + frame_center
        tag = TagDetection(6, layout.family, tuple(map(tuple, pixels)), 1, 1.)
        good = solver.solve([tag], (320, 240), None)
        self.assertTrue(good.accepted, good.rejection_reason)
        self.assertTrue(good.approximate)
        np.testing.assert_allclose(good.xy, center, atol=1e-8)
        self.assertAlmostEqual(good.yaw_deg, 37)
        self.assertAlmostEqual(good.span_m, 319 / 800)
        self.assertAlmostEqual(good.quality_metrics["corner_score"], 1)
        self.assertNotIn("height_m", good.quality_metrics)

        rotated = solver.solve([tag], (320, 240), None, ((0, -1), (1, 0)))
        self.assertAlmostEqual(rotated.yaw_deg, 127)
        self.assertAlmostEqual(rotated.span_m, 239 / 800)

        pixels[0] += (1.5, 0)
        weaker = solver.solve([replace(tag, corners_px=tuple(map(tuple, pixels)))], (320, 240), None)
        self.assertTrue(weaker.accepted, weaker.rejection_reason)
        self.assertLess(weaker.quality_metrics["corner_score"], good.quality_metrics["corner_score"])
        self.assertGreater(weaker.covariance[0][0], good.covariance[0][0])
        estimate = Estimate(True, "ok", .8, .12, good.x_m + .06, good.y_m, good.yaw_deg, .7)
        map_obs = from_map(estimate, 1, 1., .025, 12.)
        fused, _, _, _ = select_observation(map_obs, good, lambda _: True)
        fused_weaker, _, _, _ = select_observation(
            map_obs, replace(weaker, x_m=good.x_m, y_m=good.y_m), lambda _: True)
        self.assertEqual(fused.source, "FUSION")
        self.assertEqual(fused.span_m, map_obs.span_m)
        self.assertLess(fused.x_m, fused_weaker.x_m)
        self.assertLess(fused_weaker.x_m, map_obs.x_m)
        pixels[0] += (12, 0)
        skewed = solver.solve([replace(tag, corners_px=tuple(map(tuple, pixels)))], (320, 240), None)
        self.assertFalse(skewed.accepted)
        self.assertIn("planar", skewed.rejection_reason)

    def test_approximate_tag_does_not_override_a_conflicting_map(self):
        tag = self.observation(1, 1., xy=(.5, .5))
        tag = replace(tag, quality_metrics={"pose_model": "planar", "used_ids": (1, 2)})
        estimate = Estimate(True, "ok", .8, .12, 1., 1., 359., .7)
        map_obs = from_map(estimate, 1, 1., .025, 12.)
        correction, map_obs, tag_obs, reason = select_observation(
            map_obs, tag, lambda _: True, confirmed_tag_recovery=True)
        self.assertIs(correction, map_obs)
        self.assertTrue(map_obs.accepted)
        self.assertFalse(tag_obs.accepted)
        self.assertIn("conflict", reason)

    def test_live_uncalibrated_tag_updates_position_and_shows_overlay(self):
        from visualization.live_camera_viewer import LiveCameraViewer

        layout = TagMap.load(CampusMap.load())
        tag_id = 0  # Its map-aligned top edge makes expected position easy to verify.
        image = np.full((240, 320, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image[90:150, 130:190] = cv2.aruco.generateImageMarker(dictionary, tag_id, 60)[:, :, None]
        config = replace(PhysicalCameraConfig.load(), calibration_path=None)
        pipeline = LiveLocalization(NoMap(), CameraPreprocessor(config), localization_mode="apriltag",
                                    tag_localizer=AprilTagLocalizer(layout, 1.0))
        result = pipeline.process(CameraFrame(image, 1., 1, "synthetic"))
        self.assertTrue(result.tracking.measurement_used)
        np.testing.assert_allclose(result.tracking.xy_m, layout.tags[tag_id].corners.mean(axis=0), atol=.001)
        self.assertTrue(result.tag_observation.approximate)
        self.assertIn("planar", result.tag_setup)
        viewer = object.__new__(LiveCameraViewer)
        overlay = viewer._draw_tags(image, result)
        self.assertFalse(np.array_equal(overlay, image))

    def test_consistent_observations_wrap_yaw_and_use_covariance_intersection(self):
        map_obs = LocalizationObservation("MAP", 1, 1., 1.01, 1.0, 359., .4,
                                          ((.04, 0.), (0., .04)), 12.)
        tag_obs = LocalizationObservation("APRILTAG", 1, 1., 1., 1.0, 1., .4,
                                          ((.01, 0.), (0., .01)), 4.)
        out, _, _, _ = select_observation(map_obs, tag_obs, lambda _: True)
        self.assertEqual(out.source, "FUSION")
        self.assertAlmostEqual(out.covariance[0][0], .016)
        self.assertLess(abs((out.yaw_deg + 180) % 360 - 180), 2)

    def test_registration_rejects_duplicate_and_wrong_map(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tags.json"
            doc = json.loads(DEFAULT_TAG_MAP.read_text(encoding="utf-8"))
            doc["tags"].append(doc["tags"][0])
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                TagMap.load(CampusMap.load(), path)
            path.write_text(json.dumps({**doc, "reference_sha256": "different"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                TagMap.load(CampusMap.load(), path)

    def test_three_consistent_tags_and_one_wrong_placement(self):
        map_layout = TagMap.load(CampusMap.load())
        solver = AprilTagLocalizer(map_layout, physical_per_map=1.)
        center = np.array([.66, .65, .48])
        rotation = cv2.Rodrigues(np.array([.07, -.05, .1]))[0] @ np.diag([1., -1., -1.])
        K = np.array([[275., 0, 159.5], [0, 275., 119.5], [0, 0, 1.]])
        distortion = np.zeros(5)
        detections = []
        for tag_id in (20, 21, 22, 23):
            xy = map_layout.tags[tag_id].corners + ((.075, -.1) if tag_id == 23 else (0, 0))
            uv, _ = cv2.projectPoints(np.c_[xy, np.zeros(4)], cv2.Rodrigues(rotation)[0],
                                      -rotation @ center, K, distortion)
            detections.append(TagDetection(tag_id, map_layout.family, tuple(map(tuple, uv.reshape(4, 2))), 1, 0.))
        pose = solver.solve(detections, (320, 240), (K, distortion))
        self.assertTrue(pose.accepted, pose.rejection_reason)
        self.assertEqual(pose.quality_metrics["used_ids"], (20, 21, 22))
        self.assertEqual(pose.quality_metrics["dropped_ids"], (23,))
        np.testing.assert_allclose(pose.xy, center[:2], atol=1e-5)
        conflicting = solver.solve(detections[2:], (320, 240), (K, distortion))
        self.assertFalse(conflicting.accepted)

    def test_quality_rejects_tiny_unknown_boundary_and_duplicate_tags(self):
        layout = TagMap.load(CampusMap.load())
        solver = AprilTagLocalizer(layout, 1.)
        corners = ((50., 50.), (90., 50.), (90., 90.), (50., 90.))
        tag = TagDetection(0, layout.family, corners, 3, .1)
        variants = (
            (replace(tag, tag_id=100), "unknown family or ID"),
            (replace(tag, tag_family="tag25h9"), "unknown family or ID"),
            (replace(tag, corners_px=((50., 50.), (57., 50.), (57., 57.), (50., 57.))), "tag too small"),
            (replace(tag, corners_px=((0., 50.), (40., 50.), (40., 90.), (0., 90.))), "too close to image edge"),
            (replace(tag, corners_px=corners[::-1]), "invalid corner geometry"),
        )
        for variant, reason in variants:
            with self.subTest(reason=reason):
                result = solver.solve([variant], (320, 240), None)
                self.assertFalse(result.accepted)
                self.assertEqual(result.quality_metrics["rejected_tags"][variant.tag_id], reason)
        self.assertEqual(solver.solve([tag, tag], (320, 240), None).quality_metrics["rejected_tags"][0],
                         "duplicate ID in frame")

    @staticmethod
    def observation(frame_id, timestamp, xy=(.5, .5), yaw=359.):
        return LocalizationObservation("APRILTAG", frame_id, timestamp, *xy, yaw, .4,
                                       ((.0004, 0.), (0., .0004)), 4.)

    def test_tracker_initialization_prediction_and_tag_relocalization(self):
        tracker = TrackingManager(NoMap())

        def step(frame_id, timestamp, tag=None):
            return tracker.step(None, timestamp, frame_id=frame_id, tag_observation=tag, map_enabled=False)

        initial = step(1, 0., self.observation(1, 0.))
        self.assertEqual(initial.observation_source, "APRILTAG")
        rejected = step(2, .1, self.observation(2, .1, (1.1, 1.1)))
        self.assertEqual(rejected.state, TrackingState.PREDICT_ONLY)
        self.assertGreater(rejected.sigma_m, initial.sigma_m)
        self.assertIsNone(rejected.correction)
        self.assertEqual(step(3, .2, self.observation(1, 0.)).state, TrackingState.PREDICT_ONLY)
        self.assertEqual(step(4, .3).state, TrackingState.RELOCALIZING)
        # A repeatable observation in a different part of the map reinitializes after loss.
        pending = step(5, .4, self.observation(5, .4, (1.1, 1.1)))
        self.assertIn("second consistent", pending.reason)
        recovered = step(6, .5, self.observation(6, .5, (1.105, 1.1), yaw=1.))
        self.assertTrue(recovered.relocalized)
        np.testing.assert_allclose(recovered.xy_m, (1.105, 1.1))
        with self.assertRaisesRegex(ValueError, "frame IDs"):
            step(6, .6)
        self.assertEqual(step(7, .6).observation_source, "PREDICTION_ONLY")

    def test_conflicting_observations_are_not_averaged(self):
        tag = self.observation(1, 1.)
        map_obs = replace(tag, source="MAP", x_m=.8)
        correction, map_status, tag_status, reason = select_observation(map_obs, tag, lambda _: True)
        self.assertIsNone(correction)
        self.assertFalse(map_status.accepted or tag_status.accepted)
        self.assertIn("conflict", reason)
        trusted = replace(tag, quality_metrics={"used_ids": (0, 1)})
        correction, *_ = select_observation(map_obs, trusted, lambda _: True)
        self.assertEqual(correction, trusted)

    def test_map_cadence_only_skips_for_a_motion_gated_tag(self):
        calls = []

        class Matcher:
            def estimate(self, *args, **kwargs):
                calls.append(args)
                return Estimate(True, "ok", .8, .13, .5, .5, 359., .4)

        tracker = TrackingManager(Matcher())
        first = tracker.step(None, 0., frame_id=1, tag_observation=self.observation(1, 0.))
        self.assertEqual(first.observation_source, "FUSION")
        second = tracker.step(None, .1, frame_id=2, tag_observation=self.observation(2, .1), prefer_tag=True)
        self.assertEqual(second.observation_source, "APRILTAG")
        self.assertEqual(len(calls), 1)
        bad_tag = self.observation(3, .2, (1.5, 1.5))
        third = tracker.step(None, .2, frame_id=3, tag_observation=bad_tag, prefer_tag=True)
        self.assertEqual(third.observation_source, "BUILDING")
        self.assertEqual(len(calls), 2)
        tracker.step(None, .3, frame_id=4, prefer_tag=True)
        self.assertEqual(len(calls), 3)

    def test_confirmed_tag_recovery_beats_conflicting_global_match(self):
        class WrongGlobal:
            def estimate(self, *args, **kwargs):
                return Estimate(True, "ok", .9, .15, 1.8, 1.8, 359., .4)

        tracker = TrackingManager(WrongGlobal(), TrackingConfig(retry_interval_s=.05))
        tracker.step(None, 0., frame_id=1, tag_observation=self.observation(1, 0.), map_enabled=False)
        for i in range(2, 5):
            tracker.step(None, (i - 1) / 10, frame_id=i, map_enabled=False)
        pending = tracker.step(None, .4, frame_id=5, tag_observation=self.observation(5, .4, (1.1, 1.1)))
        self.assertIsNone(pending.correction)
        self.assertIn("conflicts", pending.map_observation.rejection_reason)
        recovered = tracker.step(None, .5, frame_id=6, tag_observation=self.observation(6, .5, (1.1, 1.1)))
        self.assertTrue(recovered.relocalized)
        self.assertEqual(recovered.observation_source, "APRILTAG")
        np.testing.assert_allclose(recovered.xy_m, (1.1, 1.1))

    def test_map_optical_center_alignment_and_strong_perspective(self):
        class Matcher:
            def __init__(self, offset):
                self.offset = offset

            def estimate(self, *args, **kwargs):
                return Estimate(True, "ok", .9, .15, .5 + self.offset, .5, 359., .4)

        for offset, source in ((.03, "FUSION"), (.14, "APRILTAG")):
            with self.subTest(offset=offset):
                tracker = TrackingManager(Matcher(offset))
                tag = replace(self.observation(1, 0.), quality_metrics={"image_center_offset_m": (offset, 0.)})
                result = tracker.step(None, 0., frame_id=1, tag_observation=tag)
                self.assertEqual(result.observation_source, source)
                np.testing.assert_allclose(result.xy_m, (.5, .5))
                if source == "FUSION":
                    self.assertAlmostEqual(result.map_observation.x_m, .5)
                    self.assertAlmostEqual(result.map_observation.quality_metrics["raw_image_center_xy_m"][0], .53)
                else:
                    self.assertIn("tilt", result.map_observation.rejection_reason)

    def test_same_frame_map_mask_and_tag_diagnostics(self):
        captured = []

        class MaskRecorder:
            def estimate(self, frame, *args, **kwargs):
                captured.append(kwargs["confidence_mask"])
                return Estimate(True, "ok", .75, .1, 1., 1., 350., .45)

        camera = replace(PhysicalCameraConfig.load(), size=(160, 120), rotation=90, calibration_path=None)
        layout = TagMap.load(CampusMap.load())
        pipeline = LiveLocalization(MaskRecorder(), CameraPreprocessor(camera), localization_mode="fusion",
                                    tag_localizer=AprilTagLocalizer(layout, .08))
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image = np.full((240, 320, 3), 240, np.uint8)
        image[47:107, 100:160] = cv2.aruco.generateImageMarker(dictionary, 2, 60)[:, :, None]
        first = pipeline.process(CameraFrame(image, 1., 1, "synthetic"))
        self.assertEqual(first.tracking.observation_source, "BUILDING")
        self.assertTrue(first.tag_observation.approximate)
        self.assertIn("conflict", first.tag_observation.rejection_reason)
        self.assertEqual(first.tag_detections[0].frame_id, first.map_observation.frame_id)
        self.assertEqual(first.tag_observation.timestamp, first.map_observation.timestamp)
        # Raw center (130,77) -> clockwise (162,130) -> 90×120 (61,49).
        self.assertEqual(captured[0].shape, (120, 90))
        self.assertEqual(captured[0][49, 61], 0)
        self.assertEqual(captured[0][10, 10], 1)
        pipeline.request_mode("apriltag")
        later = pipeline.process(CameraFrame(np.full_like(image, 255), 1.1, 2, "synthetic"))
        self.assertEqual(len(captured), 1)
        self.assertEqual(later.localization_mode, "apriltag")
        self.assertAlmostEqual(later.frame_aspect, 4 / 3)
        self.assertIsNone(later.tracking.xy_m)

    def test_calibration_resolution_mismatch_keeps_fusion_map_fallback(self):
        class MapFallback:
            def estimate(self, frame, *args, **kwargs):
                return Estimate(True, "ok", .8, .13, .5, .5, 359., .4)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.npz"
            K = np.array([[470., 0, 319.5], [0, 470., 239.5], [0, 0, 1.]])
            np.savez(path, camera_matrix=K, dist_coeffs=np.zeros(5), image_size=(640, 480))
            camera = replace(PhysicalCameraConfig.load(), calibration_path=path)
            layout = TagMap.load(CampusMap.load())
            pipeline = LiveLocalization(MapFallback(), CameraPreprocessor(camera), localization_mode="fusion",
                                        tag_localizer=AprilTagLocalizer(layout, .08))
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            frame = np.full((240, 320, 3), 255, np.uint8)
            frame[80:140, 100:160] = cv2.aruco.generateImageMarker(dictionary, 2, 60)[:, :, None]
            result = pipeline.process(CameraFrame(frame, 1., 1, "synthetic"))
            self.assertIn("differs from calibration", result.tag_observation.rejection_reason)
            self.assertEqual(result.tracking.observation_source, "BUILDING")
