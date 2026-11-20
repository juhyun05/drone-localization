import unittest
import math
from unittest.mock import patch

import cv2
import numpy as np

from localization.camera_interface import CameraFrame
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.illumination_correction import IlluminationConfig
from localization.image_processing import ColorConfig
from localization.lighting_pipeline import LightingProcessor
from localization.live_localization import LiveLocalization
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import Estimate, MapMatcher, RecoveryHint, SearchConfig, angle_error
from localization.paths import DATA_DIR
from localization.road_detection import detect_roads
from localization.road_map import PRINT_PREVIEW, REFERENCE_PREVIEW, RoadMap
from localization.tracking_manager import TrackingManager, TrackingState


class RoadTests(unittest.TestCase):
    def test_print_layers_have_roads_edges_centerlines_and_junctions(self):
        raster = MapRaster(CampusMap.load(), 80)
        roads = RoadMap.build(raster)
        self.assertEqual(roads.region.shape, raster.size[::-1])
        self.assertTrue(roads.from_print)
        self.assertGreater(cv2.countNonZero(roads.edges), 0)
        self.assertGreater(cv2.countNonZero(roads.centerline), 0)
        self.assertGreater(len(roads.junctions_xy_m), 0)
        self.assertLess(roads.region.mean(), 0.10)

    def test_road_detector_rejects_paper_shadow_color_and_building_overlap(self):
        image = np.full((180, 240, 3), 244, np.uint8)
        image[0:80, 160:] = 161
        image[85:96, :] = (152, 143, 133)
        image[85:150, 88:99] = (152, 143, 133)
        image[128:158, 180:210] = (40, 80, 210)
        image[100:130, 88:99] = (152, 143, 133)
        buildings = np.zeros(image.shape[:2], np.uint8)
        buildings[110:130, 88:99] = 1
        confidence = np.ones(buildings.shape, np.float32)
        confidence[:80, 160:] = 0.2
        road = detect_roads(image, buildings, confidence)
        self.assertTrue(road.region[90, 30])
        self.assertTrue(road.region[140, 92])
        self.assertFalse(road.region[115, 92])
        self.assertFalse(road.region[25, 200])
        self.assertFalse(road.region[145, 200])
        self.assertFalse(road.region[25, 25])
        self.assertGreater(road.direction_diversity, 0.08)

    def test_global_modes_refuse_blank_and_straight_road_before_search(self):
        image = np.full((180, 240, 3), 242, np.uint8)
        matcher = MapMatcher(CampusMap.load(), SearchConfig(matching_mode="adaptive", auto_scale=True))
        self.assertFalse(matcher.estimate(image).success)
        image[88:97] = (152, 143, 133)
        estimate = matcher.estimate(image)
        self.assertFalse(estimate.success)
        self.assertIn("road geometry", estimate.reason)
        self.assertGreater(estimate.road_coverage, 0.02)

    def test_north_up_print_intersection_recovers_position_yaw_span_and_debug_projection(self):
        campus = CampusMap.load()
        config = CameraConfig(height_m=0.28)
        camera = SyntheticCamera(MapRaster(campus, 800), config)
        path = PRINT_PREVIEW if PRINT_PREVIEW.is_file() else REFERENCE_PREVIEW
        reference = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        camera.scene = cv2.resize(reference, camera.raster.size)
        truth = Pose(1.598, 0.59, 96.0)
        frame = camera.capture(truth)
        search = SearchConfig(auto_scale=True, nominal_span_m=config.footprint_m)
        building = MapMatcher(campus, search).estimate(frame)
        self.assertFalse(building.success)

        class Identity:
            def apply(self, frame):
                return frame

        matcher = MapMatcher(campus, SearchConfig(auto_scale=True, nominal_span_m=config.footprint_m,
                                                matching_mode="adaptive"))
        result = LiveLocalization(matcher, Identity(), global_only=True, road_debug=True).process(
            CameraFrame(frame, 0, 1, "synthetic"))
        match = result.measurement
        self.assertTrue(match.success, match)
        self.assertEqual(match.observation_source, "ROAD_ONLY")
        self.assertLess(math.hypot(match.x_m - truth.x_m, match.y_m - truth.y_m), 0.025)
        self.assertLess(angle_error(match.yaw_deg, truth.yaw_deg), 8)
        self.assertLess(abs(match.span_m - config.footprint_m), 0.04)
        self.assertEqual(result.diagnostic.shape, (360, 720, 3))
        self.assertGreater(cv2.countNonZero(matcher.project_roads(frame.shape, match)), 0)

    def test_xiao_gate_relocalizes_without_prior_when_paper_edges_are_visible(self):
        path = DATA_DIR / "fixtures" / "xiao_main_gate.jpg"
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        matcher = MapMatcher(CampusMap.load(), SearchConfig(auto_scale=True, matching_mode="adaptive"),
                             ColorConfig(saturation_tolerance=78, hue_shift=3, hue_band_width=15))
        localizer = LiveLocalization(
            matcher, CameraPreprocessor(PhysicalCameraConfig.load()), global_only=True,
            lighting=LightingProcessor(IlluminationConfig("combined"), shadow_aware=True),
        )
        match = localizer.process(CameraFrame(frame, 0, 1, "xiao")).measurement
        self.assertTrue(match.success, match)
        self.assertEqual(match.observation_source, "ROAD_ONLY")
        self.assertGreaterEqual(match.score, 0.58)
        self.assertLess(math.hypot(match.x_m - 1.40, match.y_m - 0.86), 0.05)
        self.assertLess(angle_error(match.yaw_deg, 270), 8)

    def test_same_gate_view_recovers_with_local_correlations_after_input_gap(self):
        path = DATA_DIR / "fixtures" / "xiao_main_gate.jpg"
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        matcher = MapMatcher(CampusMap.load(), SearchConfig(auto_scale=True, matching_mode="adaptive"),
                             ColorConfig(saturation_tolerance=78, hue_shift=3, hue_band_width=15))
        localizer = LiveLocalization(
            matcher, CameraPreprocessor(PhysicalCameraConfig.load()),
            lighting=LightingProcessor(IlluminationConfig("combined"), shadow_aware=True),
        )
        initial = localizer.process(CameraFrame(image, 0, 1, "test", connection=1, input_epoch=0))
        self.assertTrue(initial.tracking.measurement_used)
        self.assertEqual(initial.match_mode, "global")

        with patch.object(cv2, "matchTemplate", wraps=cv2.matchTemplate) as correlate:
            recovered = localizer.process(CameraFrame(image, 2.4, 2, "test", connection=1, input_epoch=1))
        self.assertTrue(recovered.tracking.measurement_used)
        self.assertEqual(recovered.match_mode, "local")
        self.assertLessEqual(correlate.call_count, 27)
        self.assertEqual(recovered.tracking.velocity_mps, (0, 0))

        reconnected = localizer.process(CameraFrame(image, 4.8, 3, "test", connection=2, input_epoch=1))
        self.assertTrue(reconnected.measurement.recovered_locally)
        self.assertEqual(reconnected.match_mode, "local")

    def test_reconnect_searches_wider_local_region_then_falls_back_globally(self):
        campus = CampusMap.load()
        camera = SyntheticCamera(MapRaster(campus, 320), CameraConfig())
        matcher = MapMatcher(campus, SearchConfig(auto_scale=True, matching_mode="adaptive"))
        localizer = LiveLocalization(matcher, CameraPreprocessor(PhysicalCameraConfig.load()))
        poses = (Pose(1.73, 2.02, 37), Pose(1.90, 2.04, 49), Pose(0.83, 0.82, 124))
        previous = None
        for i, pose in enumerate(poses):
            frame = CameraFrame(camera.capture(pose), 2.4 * i, i + 1, "test", connection=i + 1, input_epoch=0)
            result = localizer.process(frame)
            self.assertTrue(result.tracking.measurement_used, result)
            expected = "local" if i == 1 else "global"
            self.assertEqual(result.match_mode, expected)
            self.assertLess(math.dist(result.tracking.xy_m, (pose.x_m, pose.y_m)), 0.05)
            if i == 1:
                self.assertTrue(result.measurement.recovered_locally)
                self.assertFalse(result.measurement.tracking_window)
                previous = result.measurement

        fallback = matcher.estimate(camera.capture(poses[-1]), recovery_hint=RecoveryHint(previous, reconnected=True))
        self.assertTrue(fallback.success, fallback)
        self.assertFalse(fallback.recovered_locally)
        self.assertLess(math.dist((fallback.x_m, fallback.y_m), (poses[-1].x_m, poses[-1].y_m)), 0.05)

    def test_prediction_expiry_grows_uncertainty_and_allows_road_reacquisition(self):
        class Matcher:
            def estimate(self, frame, *args, **kwargs):
                if frame == "blank":
                    return Estimate(False, "insufficient reliable road geometry or buildings")
                return Estimate(True, "matched", 0.9, 0.2, 1.2, 0.9, 15, 0.55, frame)

        tracker = TrackingManager(Matcher())
        building = tracker.step("BUILDING", 0.0)
        road = tracker.step("ROAD_ONLY", 0.1)
        prediction = tracker.step("blank", 0.3)
        self.assertEqual(building.observation_source, "BUILDING")
        self.assertEqual(road.observation_source, "ROAD_ONLY")
        self.assertEqual(prediction.observation_source, "PREDICTION_ONLY")
        self.assertFalse(prediction.measurement_used)
        expired = tracker.step("blank", 1.5)
        self.assertEqual(expired.state, TrackingState.RELOCALIZING)
        self.assertIsNone(expired.xy_m)
        self.assertGreater(expired.sigma_m, prediction.sigma_m)
        recovered = tracker.step("ROAD_ONLY", 2.2)
        self.assertTrue(recovered.relocalized)
        self.assertTrue(recovered.measurement_used)

    def test_separated_repeated_curves_do_not_localize_globally(self):
        roads = []
        for offset in (0.0, 1.2):
            roads.append({
                "display_color": "#858f98", "display_linewidth_pt": 10.5,
                "centerline_xy_m": [(offset + 0.27, 0.70), (offset + 0.44, 0.70),
                                    (offset + 0.51, 0.75), (offset + 0.56, 0.88),
                                    (offset + 0.56, 1.00)],
            })
        buildings = [{"color": "#54cc55",
                      "polygon_xy_m": [(2.1, 2.1), (2.2, 2.1), (2.2, 2.2), (2.1, 2.2)]}]
        campus = CampusMap(2.5, 2.5, buildings, roads, {})
        camera = SyntheticCamera(MapRaster(campus, 240), CameraConfig(height_m=0.29))
        frame = camera.capture(Pose(0.48, 0.79, 0))
        matcher = MapMatcher(campus, SearchConfig(nominal_span_m=camera.config.footprint_m,
                                                  auto_scale=True, matching_mode="road"))
        estimate = matcher.estimate(frame)
        self.assertFalse(estimate.success)
        self.assertEqual(estimate.reason, "ambiguous pose")
        self.assertGreater(estimate.score, 0.58)


if __name__ == "__main__":
    unittest.main()
