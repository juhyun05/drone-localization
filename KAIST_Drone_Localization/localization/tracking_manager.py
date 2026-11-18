"""Use only BGR frames, capture times, and optional externally supplied priors."""

from dataclasses import dataclass
from enum import Enum
import math
from time import perf_counter

from .kalman_filter import FilterConfig, PositionFilter
from .map_matching import Estimate, MapMatcher


class TrackingState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    GLOBAL_SEARCH = "GLOBAL_SEARCH"
    TRACKING = "TRACKING"
    PREDICT_ONLY = "PREDICT_ONLY"
    RELOCALIZING = "RELOCALIZING"


@dataclass(frozen=True)
class TrackingConfig:
    max_misses: int = 3
    retry_interval_s: float = 0.6
    max_sigma_m: float = 0.28
    min_radius_m: float = 0.13
    max_radius_m: float = 0.40
    yaw_radius_deg: float = 25.0
    yaw_growth_deg_s: float = 35.0
    innovation_limit: float = 9.21
    max_speed_mps: float = 0.9
    position_slack_m: float = 0.06
    max_prediction_s: float = 1.2

    def __post_init__(self):
        values = (self.retry_interval_s, self.max_sigma_m, self.min_radius_m, self.max_radius_m,
                  self.yaw_radius_deg, self.yaw_growth_deg_s, self.innovation_limit,
                  self.max_speed_mps, self.position_slack_m, self.max_prediction_s)
        if self.max_misses < 1 or not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("tracking limits must be positive and finite")
        if self.max_radius_m < self.min_radius_m:
            raise ValueError("max_radius_m must be at least min_radius_m")


@dataclass(frozen=True)
class TrackingResult:
    timestamp_s: float
    state: TrackingState
    attempt: str
    measurement: Estimate | None
    measurement_used: bool
    reason: str
    xy_m: tuple[float, float] | None
    velocity_mps: tuple[float, float] | None
    yaw_deg: float | None
    sigma_m: float | None
    relocalized: bool
    match_ms: float
    filter_ms: float
    total_ms: float
    span_m: float | None = None
    observation_source: str | None = None


class TrackingManager:
    def __init__(self, matcher: MapMatcher, config: TrackingConfig = TrackingConfig(),
                 filter_config: FilterConfig = FilterConfig(), *, initial_xy=None):
        self.matcher = matcher
        self.config = config
        self.filter_config = filter_config
        self.filter = PositionFilter(initial_xy, filter_config) if initial_xy is not None else None
        self.state = TrackingState.UNINITIALIZED
        self.timestamp = None
        self.started_at = None
        self.last_global = None
        self.last_observed = None
        self.yaw = None
        self.span = None
        self.misses = 0

    def step(self, frame, timestamp_s: float, *, confidence_mask=None,
             timings=None, on_labels=None, on_roads=None) -> TrackingResult:
        """Process one frame at a strictly increasing capture timestamp in seconds."""
        started = perf_counter()
        if not math.isfinite(timestamp_s) or (self.timestamp is not None and timestamp_s <= self.timestamp):
            raise ValueError("capture timestamps must be finite and strictly increasing")
        previous_time, self.timestamp = self.timestamp, timestamp_s
        if self.started_at is None:
            self.started_at = timestamp_s
        filter_time = perf_counter()
        if self.filter is not None and previous_time is not None:
            self.filter.predict(timestamp_s - previous_time)
        filter_ms = (perf_counter() - filter_time) * 1000

        relocalizing = self.state == TrackingState.RELOCALIZING
        attempt = "local"
        measurement = None
        search_start = perf_counter()
        options = {} if confidence_mask is None else {"confidence_mask": confidence_mask}
        if timings is not None:
            options["timings"] = timings
        if on_labels is not None:
            options["on_labels"] = on_labels
        if on_roads is not None:
            options["on_roads"] = on_roads
        if self.filter is None or relocalizing:
            if self.last_global is None or timestamp_s - self.last_global >= self.config.retry_interval_s - 1e-9:
                attempt = "global"
                self.last_global = timestamp_s
                measurement = self.matcher.estimate(frame, **options)
            else:
                attempt = "none"
        else:
            sigma = self.filter.position_sigma_m
            radius = min(self.config.max_radius_m, max(self.config.min_radius_m, 3 * sigma))
            age = 0 if self.last_observed is None else timestamp_s - self.last_observed[0]
            yaw_window = self.config.yaw_radius_deg + self.config.yaw_growth_deg_s * age
            # A miss releases the width prior. The auto matcher then searches its full
            # span range locally before the state machine releases the position prior.
            measurement = self.matcher.estimate(
                frame, self.filter.position, radius_m=radius, prior_yaw=self.yaw,
                yaw_radius_deg=yaw_window, prior_span_m=self.span if self.misses == 0 else None,
                **options,
            )
        match_ms = (perf_counter() - search_start) * 1000 if measurement is not None else 0.0

        filter_time = perf_counter()
        reason = measurement.reason if measurement is not None else "waiting for global retry"
        used = False
        if measurement is not None and measurement.success:
            xy = (measurement.x_m, measurement.y_m)
            if attempt == "global":
                self.filter = PositionFilter(xy, self.filter_config)
                used = True
            elif self._plausible(xy, timestamp_s):
                self.filter.update(xy)
                used = True
            else:
                reason = "measurement rejected by motion gate"

        if used:
            self.state = TrackingState.TRACKING
            self.last_observed = timestamp_s, self.filter.position
            self.yaw, self.span = measurement.yaw_deg, measurement.span_m
            self.misses = 0
        elif self.filter is None:
            self.state = TrackingState.GLOBAL_SEARCH
        else:
            self.misses += 1
            lost = self.misses >= self.config.max_misses or self.filter.position_sigma_m >= self.config.max_sigma_m
            self.state = TrackingState.RELOCALIZING if relocalizing or lost else TrackingState.PREDICT_ONLY

        xy = self.filter.position if self.filter else None
        velocity = self.filter.velocity if self.filter else None
        sigma = self.filter.position_sigma_m if self.filter else None
        yaw, span = self.yaw, self.span
        source = None
        if used:
            source = measurement.observation_source or "BUILDING"
        elif xy is not None:
            observed_at = self.started_at if self.last_observed is None else self.last_observed[0]
            if sigma >= self.config.max_sigma_m or timestamp_s - observed_at >= self.config.max_prediction_s:
                xy = velocity = yaw = span = None
                self.state = TrackingState.RELOCALIZING
                reason += " / prediction expired"
            else:
                source = "PREDICTION_ONLY"
        filter_ms += (perf_counter() - filter_time) * 1000
        return TrackingResult(timestamp_s, self.state, attempt, measurement, used, reason, xy, velocity,
                              yaw, sigma, relocalizing and used, match_ms, filter_ms,
                              (perf_counter() - started) * 1000, span, source)

    def _plausible(self, xy, timestamp_s):
        if self.filter.innovation_squared(xy) > self.config.innovation_limit:
            return False
        if self.last_observed is None:
            return True
        observed_time, position = self.last_observed
        limit = self.config.position_slack_m + self.config.max_speed_mps * (timestamp_s - observed_time)
        return math.dist(xy, position) <= limit
