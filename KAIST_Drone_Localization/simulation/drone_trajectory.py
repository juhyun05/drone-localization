"""Timestamped model-meter trajectories for the synthetic camera."""

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np

from localization.camera_simulator import Pose, SyntheticCamera


@dataclass(frozen=True)
class DroneSample:
    timestamp_s: float
    pose: Pose
    vx_mps: float
    vy_mps: float


@dataclass(frozen=True)
class DroneTrajectory:
    duration_s: float
    motion: Callable[[float], tuple[np.ndarray, np.ndarray]]

    @classmethod
    def waypoints(cls, points, speed_mps: float):
        """Traverse straight segments at constant speed with instantaneous corners."""
        xy = np.asarray(points, dtype=float)
        if (xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all()
                or not math.isfinite(speed_mps) or speed_mps <= 0):
            raise ValueError("provide at least two finite (x,y) waypoints and a positive speed")
        delta = np.diff(xy, axis=0)
        length = np.linalg.norm(delta, axis=1)
        if np.any(length == 0):
            raise ValueError("consecutive waypoints must differ")
        velocity = delta / length[:, None] * speed_mps
        times = np.r_[0.0, np.cumsum(length / speed_mps)]

        def motion(t):
            if t >= times[-1]:
                return xy[-1], np.zeros(2)
            i = np.searchsorted(times, t, side="right") - 1
            return xy[i] + velocity[i] * (t - times[i]), velocity[i]

        return cls(float(times[-1]), motion)

    @classmethod
    def circle(cls, center, radius_m: float, speed_mps: float):
        """Complete one counterclockwise circle starting east of its center."""
        if not all(math.isfinite(v) and v > 0 for v in (radius_m, speed_mps)):
            raise ValueError("circle radius and speed must be positive and finite")
        origin = np.asarray(center, dtype=float)
        omega = speed_mps / radius_m

        def motion(t):
            angle = omega * t
            cos, sin = math.cos(angle), math.sin(angle)
            return origin + radius_m * np.array([cos, sin]), speed_mps * np.array([-sin, cos])

        return cls(2 * math.pi / omega, motion)

    @classmethod
    def preset(cls, name: str, speed_mps: float):
        """Return a line, square, or circle within the campus model."""
        if name == "line":
            return cls.waypoints([(1.65, 2.08), (2.25, 1.77)], speed_mps)
        if name == "square":
            return cls.waypoints([(1.72, 1.77), (2.07, 1.77), (2.07, 2.12),
                                  (1.72, 2.12), (1.72, 1.77)], speed_mps)
        if name == "circle":
            return cls.circle((0.88, 0.83), 0.18, speed_mps)
        raise ValueError(f"unknown trajectory {name!r}")

    def samples(self, dt_s: float, yaw_deg: float = 37, yaw_rate_deg_s: float = 12):
        """Sample the path in seconds, including the endpoint, with independent camera yaw."""
        if not all(math.isfinite(v) for v in (dt_s, yaw_deg, yaw_rate_deg_s)) or dt_s <= 0:
            raise ValueError("sample interval must be positive and yaw settings finite")
        times = np.arange(0, self.duration_s, dt_s)
        times = np.r_[times[times < self.duration_s - 1e-9], self.duration_s]
        samples = []
        for t in times:
            xy, velocity = self.motion(t)
            pose = Pose(float(xy[0]), float(xy[1]), (yaw_deg + yaw_rate_deg_s * t) % 360)
            samples.append(DroneSample(float(t), pose, float(velocity[0]), float(velocity[1])))
        return samples


def capture_sample(camera: SyntheticCamera, sample: DroneSample):
    """Return only the BGR image and timestamp consumed by the tracker."""
    return camera.capture(sample.pose), sample.timestamp_s
