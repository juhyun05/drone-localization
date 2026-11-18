import sys
import unittest
from pathlib import Path

import numpy as np

if __name__ == "__main__" and not __package__:
    # Direct script execution puts tests/, rather than the project root, on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization.kalman_filter import PositionFilter
from localization.map_matching import Estimate
from localization.tracking_manager import TrackingConfig, TrackingManager, TrackingState
from simulation.drone_trajectory import DroneTrajectory
from simulation.playback import latest_due_index


class TrackingTests(unittest.TestCase):
    def test_waypoint_speed_and_timestamp(self):
        path = DroneTrajectory.waypoints([(0, 0), (1, 0), (1, 1)], speed_mps=0.5)
        samples = path.samples(0.5)
        self.assertEqual(len(samples), 9)
        self.assertEqual(samples[4].timestamp_s, 2)
        self.assertEqual(samples[4].pose.y_m, 0)
        self.assertEqual(samples[4].vy_mps, 0.5)
        self.assertEqual(samples[-1].pose.y_m, 1)

    def test_predict_grows_covariance_and_uses_timestamp_gap(self):
        kf = PositionFilter((0, 0))
        kf.predict(1)
        kf.update((0.4, 0))
        x, vx = kf.position[0], kf.velocity[0]
        variance = kf.covariance[0, 0]
        kf.predict(2)
        self.assertAlmostEqual(kf.position[0], x + 2 * vx)
        self.assertGreater(kf.covariance[0, 0], variance)

    def test_latest_frame_drops_backlog(self):
        timestamps = [0, 0.1, 0.2, 0.3, 0.4]
        self.assertEqual(latest_due_index(timestamps, 0.35, 0), 3)
        self.assertIsNone(latest_due_index(timestamps, 0.35, 3))
        self.assertEqual(latest_due_index(timestamps, 10, 3), 4)

    def test_outlier_is_gated_then_global_can_reacquire(self):
        def match(x):
            return Estimate(True, "matched", 0.95, 0.5, x, 1.0, 0, 0.67)

        class ScriptedMatcher:
            responses = iter([match(1), match(2), Estimate(False, "blank"), match(2)])
            priors = []

            def estimate(self, frame, prior_xy=None, **kwargs):
                self.priors.append(prior_xy)
                return next(self.responses)

        matcher = ScriptedMatcher()
        tracker = TrackingManager(matcher, TrackingConfig(max_misses=2))
        frame = np.zeros((2, 2, 3), np.uint8)
        self.assertEqual(tracker.state, TrackingState.UNINITIALIZED)
        self.assertEqual(tracker.step(frame, 0).state, TrackingState.TRACKING)
        rejected = tracker.step(frame, 0.2)
        self.assertEqual(rejected.state, TrackingState.PREDICT_ONLY)
        self.assertFalse(rejected.measurement_used)
        self.assertEqual(rejected.xy_m, (1, 1))
        self.assertEqual(tracker.step(frame, 0.4).state, TrackingState.RELOCALIZING)
        recovered = tracker.step(frame, 0.6)
        self.assertEqual(recovered.state, TrackingState.TRACKING)
        self.assertTrue(recovered.relocalized)
        self.assertEqual(recovered.xy_m, (2, 1))
        self.assertEqual(matcher.priors, [None, (1, 1), (1, 1), None])
        with self.assertRaises(ValueError):
            tracker.step(frame, 0.6)


if __name__ == "__main__":
    unittest.main()
