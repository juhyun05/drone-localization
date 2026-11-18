import math
import sys
import unittest
from pathlib import Path

import numpy as np
import cv2

if __name__ == "__main__" and not __package__:
    # Direct script execution puts tests/, rather than the project root, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.image_processing import BuildingDetector
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import MapMatcher, SearchConfig, _color_iou, angle_error
from benchmark import projected_frame


class LocalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campus = CampusMap.load()
        cls.raster = MapRaster(cls.campus, 320)
        cls.camera = SyntheticCamera(cls.raster, CameraConfig())
        cls.matcher = MapMatcher(cls.campus, SearchConfig())

    def test_campus_corners_have_north_up(self):
        np.testing.assert_allclose(self.raster.xy_to_uv([[0, 0], [2.5, 2.5]]),
                                   [[0, 800], [800, 0]])
        np.testing.assert_allclose(self.raster.uv_to_xy([[0, 800], [800, 0]]),
                                   [[0, 0], [2.5, 2.5]])

    def test_rotated_frame_localizes_from_pixels(self):
        pose = Pose(1.73, 2.02, 37)
        result = self.matcher.estimate(self.camera.capture(pose))
        self.assertTrue(result.success, result)
        self.assertLess(math.hypot(result.x_m - pose.x_m, result.y_m - pose.y_m), 0.04)
        self.assertLess(angle_error(result.yaw_deg, pose.yaw_deg), 7)

    def test_local_search_checks_near_previous_position(self):
        pose = Pose(1.74, 2.00, 44)
        result = self.matcher.estimate(self.camera.capture(pose), prior_xy=(1.7, 1.96))
        self.assertTrue(result.success, result)
        self.assertLess(math.hypot(result.x_m - pose.x_m, result.y_m - pose.y_m), 0.04)

    def test_empty_frame_returns_no_pose(self):
        frame = np.full((180, 240, 3), 230, np.uint8)
        result = self.matcher.estimate(frame)
        self.assertFalse(result.success)
        self.assertIsNone(result.x_m)

    def test_label_only_detection_matches_full_observation(self):
        detector = BuildingDetector(self.campus.palette)
        frame = self.camera.capture(Pose(1.73, 2.02, 37))
        full = detector.detect(frame)
        labels_only = detector.detect(frame, include_contours=False)
        self.assertTrue(full.contours)
        self.assertFalse(labels_only.contours)
        np.testing.assert_array_equal(labels_only.labels, full.labels)

    def test_cropped_iou_matches_direct_pixel_counts(self):
        mapped = np.zeros((11, 13), np.uint8)
        mapped[1:3, 2:4] = 1
        mapped[6:9, 7:11] = 2
        visible = np.ones((3, 3), np.float32)
        visible[0, 0] = 0
        for color in (1, 2, 3):
            with self.subTest(color=color):
                rotated = np.zeros((5, 5), np.uint8)
                rotated[2, 2:4] = color
                mask = (mapped == color).astype(np.uint8)
                scores = _color_iou((mapped > 0).astype(np.float32),
                                    {color: (mask, cv2.boundingRect(mask))},
                                    rotated, (1, 1, visible), np.count_nonzero(rotated))
                observed = rotated[1:4, 1:4] > 0
                expected = np.empty_like(scores)
                for row, col in np.ndindex(expected.shape):
                    patch = mapped[row + 1:row + 4, col + 1:col + 4]
                    overlap = np.count_nonzero(observed & (patch == color))
                    union = np.count_nonzero((patch > 0) & (visible > 0)) + 2 - overlap
                    expected[row, col] = overlap / union
                np.testing.assert_array_equal(scores, expected)

    def test_vector_projection_with_scale_noise_and_lighting(self):
        pose = Pose(0.83, 0.82, 124)
        camera = CameraConfig(height_m=0.44, noise_std=4, brightness=0.87,
                              channel_gains=(1.02, 0.99, 1.01))
        frame = projected_frame(self.campus, pose, camera, seed=43)
        result = self.matcher.estimate(frame)
        self.assertTrue(result.success, result)
        self.assertLess(math.hypot(result.x_m - pose.x_m, result.y_m - pose.y_m), 0.05)
        self.assertLess(angle_error(result.yaw_deg, pose.yaw_deg), 10)

    def test_very_sparse_scaled_view_does_not_report_a_pose(self):
        frame = projected_frame(self.campus, Pose(1.53, 0.77, 67),
                                CameraConfig(height_m=0.595), seed=46)
        result = self.matcher.estimate(frame)
        self.assertFalse(result.success, result)
        self.assertIsNone(result.x_m)

    def test_two_identical_landmarks_are_ambiguous(self):
        def building(x):
            return {"color": "#ee7777", "polygon_xy_m": [
                [x, 0.9], [x + 0.18, 0.9], [x + 0.18, 1.1],
                [x + 0.08, 1.1], [x + 0.08, 1.0], [x, 1.0],
            ]}

        campus = CampusMap(2.5, 2.5, [building(0.4), building(1.65)], [], {})
        camera = SyntheticCamera(MapRaster(campus, 320), CameraConfig())
        matcher = MapMatcher(campus, SearchConfig())
        result = matcher.estimate(camera.capture(Pose(0.50, 1, 0)))
        self.assertFalse(result.success, result)
        self.assertEqual(result.reason, "ambiguous pose")


if __name__ == "__main__":
    unittest.main()
