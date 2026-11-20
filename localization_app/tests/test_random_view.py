import sys
import json
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

import numpy as np

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization.camera_simulator import CameraConfig, Pose
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import Estimate
from localization.random_view import BlurConfig, RandomViewExperiment, score_estimate


class RecordingMatcher:
    def __init__(self):
        self.frames = []

    def estimate(self, frame):
        self.frames.append(frame)
        return Estimate(True, "matched", 0.8, 0.3, 1.0, 1.0, 359)


class RandomViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raster = MapRaster(CampusMap.load(), 160)

    def experiment(self, **options):
        return RandomViewExperiment(self.raster, CameraConfig(noise_std=3), RecordingMatcher(),
                                    seed=42, **options)

    def test_seed_repeats_bounded_pose_and_noisy_frame(self):
        experiment = self.experiment()
        first = experiment.sample()
        experiment.reseed(42)
        second = experiment.sample()
        self.assertEqual(first.pose, second.pose)
        self.assertTrue(0 <= first.pose.x_m <= 2.5 and 0 <= first.pose.y_m <= 2.5)
        self.assertTrue(0 <= first.pose.yaw_deg < 360)
        self.assertGreaterEqual(first.coverage, 0.05)
        np.testing.assert_array_equal(first.frame, second.frame)

    def test_rejection_sampling_is_bounded_and_zero_coverage_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "3 attempts"):
            self.experiment(min_building_coverage=1, max_attempts=3).sample()
        self.assertEqual(self.experiment(min_building_coverage=0).sample().attempts, 1)

    def test_blur_validation(self):
        for kernel, sigma in ((0, 2), (8, 2), (-3, 2), (9, -1), (9, float("nan"))):
            with self.subTest(kernel=kernel, sigma=sigma), self.assertRaises(ValueError):
                BlurConfig(kernel, sigma)

    def test_matching_is_independent_and_reblur_keeps_original(self):
        experiment = self.experiment()
        result = experiment.run(BlurConfig())
        self.assertEqual(result.sample.frame.shape, result.blurred_frame.shape)
        self.assertFalse(np.array_equal(result.sample.frame, result.blurred_frame))
        self.assertIs(experiment.matcher.frames[0], result.sample.frame)
        self.assertIs(experiment.matcher.frames[1], result.blurred_frame)
        updated = experiment.reblur(result, BlurConfig(15, 3))
        self.assertIs(updated.sample, result.sample)
        self.assertIs(updated.original, result.original)
        self.assertIs(experiment.matcher.frames[2], updated.blurred_frame)
        self.assertEqual(len(experiment.matcher.frames), 3)

    def test_position_yaw_and_failure_errors(self):
        truth = Pose(0, 0, 1)
        scored = score_estimate(Estimate(True, "matched", x_m=0.3, y_m=0.4, yaw_deg=359), truth, 12)
        self.assertAlmostEqual(scored.position_error_m, 0.5)
        self.assertEqual(scored.yaw_error_deg, 2)
        failure = score_estimate(Estimate(False, "ambiguous pose"), truth, 12)
        self.assertIsNone(failure.position_error_m)
        self.assertIsNone(failure.yaw_error_deg)

    def test_viewer_draws_three_poses_and_exports_pair(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tests.random_view_test import RandomViewViewer

        experiment = self.experiment()
        result = experiment.run()
        viewer = RandomViewViewer(experiment, BlurConfig(), interactive=False)
        try:
            viewer.show_result(result)
            self.assertEqual(viewer.map_axes.get_legend_handles_labels()[1],
                             ["Camera footprint", "Ground truth", "Original estimate", "Blurred estimate"])
            self.assertEqual(viewer.original_axes.get_title(), "Original Camera View")
            self.assertEqual(viewer.blurred_axes.get_title(), "Blurred Camera View")
            xmin, xmax = viewer.map_axes.get_xlim()
            ymin, ymax = viewer.map_axes.get_ylim()
            for x, y in result.sample.footprint_xy_m:
                self.assertTrue(xmin < x < xmax and ymin < y < ymax)
            with TemporaryDirectory() as temp:
                path = viewer.save(Path(temp) / "pair.png")
                self.assertGreater(path.stat().st_size, 0)
                record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual(record["ground_truth"]["yaw_deg"], result.sample.pose.yaw_deg)
                self.assertTrue(record["original"]["estimate"]["success"])
        finally:
            plt.close(viewer.figure)


if __name__ == "__main__":
    unittest.main()
