"""Constant-velocity Kalman filter with a timestamp-dependent process covariance."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    acceleration_density: float = 0.025  # white acceleration spectral density (m²/s³)
    measurement_sigma_m: float = 0.025
    initial_position_sigma_m: float = 0.035
    initial_velocity_sigma_mps: float = 0.30

    def __post_init__(self):
        values = (self.acceleration_density, self.measurement_sigma_m,
                  self.initial_position_sigma_m, self.initial_velocity_sigma_mps)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("filter noise settings must be positive and finite")


class PositionFilter:
    def __init__(self, xy, config: FilterConfig = FilterConfig()):
        self.config = config
        self.state = np.array([*xy, 0.0, 0.0], dtype=float)
        self.covariance = np.diag([config.initial_position_sigma_m ** 2] * 2
                                  + [config.initial_velocity_sigma_mps ** 2] * 2)
        self.r = np.eye(2) * config.measurement_sigma_m ** 2

    @property
    def position(self):
        """Current (x, y) in meters."""
        return tuple(self.state[:2])

    @property
    def velocity(self):
        """Current (vx, vy) in meters per second."""
        return tuple(self.state[2:])

    @property
    def position_sigma_m(self):
        """Largest principal standard deviation of the position covariance."""
        return math.sqrt(max(0.0, np.linalg.eigvalsh(self.covariance[:2, :2])[-1]))

    def predict(self, dt_s: float):
        """Propagate motion and covariance through elapsed capture time in seconds."""
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("prediction interval must be positive and finite")
        eye = np.eye(2)
        f = np.eye(4)
        f[:2, 2:] = dt_s * eye
        q = self.config.acceleration_density * np.block([
            [dt_s ** 3 / 3 * eye, dt_s ** 2 / 2 * eye],
            [dt_s ** 2 / 2 * eye, dt_s * eye],
        ])
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q

    def innovation_squared(self, xy):
        """Return normalized squared innovation for a candidate position measurement."""
        residual = np.asarray(xy) - self.state[:2]
        s = self.covariance[:2, :2] + self.r
        return float(residual @ np.linalg.solve(s, residual))

    def update(self, xy):
        """Fuse an accepted (x,y) measurement, with H selecting the first two states."""
        s = self.covariance[:2, :2] + self.r
        gain = np.linalg.solve(s, self.covariance[:2]).T
        self.state += gain @ (np.asarray(xy) - self.state[:2])
        joseph = np.eye(4)
        joseph[:, :2] -= gain
        covariance = joseph @ self.covariance @ joseph.T + gain @ self.r @ gain.T
        self.covariance = (covariance + covariance.T) / 2
