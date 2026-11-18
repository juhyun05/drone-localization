"""Adapt timestamped frames to the existing global matcher or tracker."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .camera_interface import INPUT_GAP_S
from .lighting_pipeline import diagnostic_panel
from .map_matching import Estimate
from .road_debug import road_panel, road_views
from .road_detection import detect_roads
from .tracking_manager import TrackingManager, TrackingResult


@dataclass(frozen=True)
class LightingStats:
    method: str
    shadow_aware: bool
    effective_coverage: float | None
    shadow_ratio: float | None
    preprocess_ms: float
    match_ms: float


@dataclass(frozen=True)
class LiveResult:
    measurement: Estimate | None
    tracking: TrackingResult | None
    processing_ms: float
    frame_aspect: float | None = None
    lighting: LightingStats | None = None
    diagnostic: np.ndarray | None = None
    preprocessing_ms: float | None = None
    lighting_ms: float | None = None
    hsv_ms: float | None = None
    match_mode: str = "none"
    matching_ms: float | None = None
    tracking_ms: float | None = None
    localization_ms: float | None = None
    diagnostic_ms: float | None = None
    projected_xy_m: tuple[float, float] | None = None
    projected_timestamp: float | None = None
    road_ms: float | None = None


class LiveLocalization:
    def __init__(self, matcher, preprocess, *, global_only=False, stale_s=INPUT_GAP_S, lighting=None,
                 road_debug=False):
        self.matcher = matcher
        self.preprocess = preprocess
        self.global_only = global_only
        self.stale_s = stale_s
        self.tracker = None
        self.last_input = None
        self.lighting = lighting
        self.road_debug = road_debug

    def process(self, frame):
        start = perf_counter()
        original = self.preprocess.apply(frame.image)
        preprocessed = perf_counter()
        observation = self.lighting.apply(original) if self.lighting is not None else None
        corrected = perf_counter()
        image = original if observation is None else observation.image
        aspect = image.shape[0] / image.shape[1]
        confidence = None if observation is None else observation.confidence
        search_start = perf_counter()
        tracking = None
        timings = {}
        debug_labels, debug_roads = [], []
        options = {}
        if self.road_debug or (self.lighting is not None and self.lighting.debug):
            options["on_labels"] = debug_labels.append
        if self.road_debug:
            options["on_roads"] = debug_roads.append
        if self.global_only:
            if confidence is not None:
                options["confidence_mask"] = confidence
            measurement = self.matcher.estimate(image, timings=timings, **options)
            mode = "global"
            match_total = (perf_counter() - search_start) * 1000
        else:
            key = frame.connection, frame.image.shape, frame.input_epoch
            if (self.last_input is None or key != self.last_input[0]
                    or (frame.input_epoch is None and frame.timestamp - self.last_input[1] >= self.stale_s)):
                self.tracker = TrackingManager(self.matcher)
            self.last_input = key, frame.timestamp
            tracking = self.tracker.step(image, frame.timestamp, confidence_mask=confidence,
                                         timings=timings, **options)
            measurement = tracking.measurement
            mode, match_total = tracking.attempt, tracking.match_ms
        elapsed = perf_counter()
        hsv_ms = timings.get("hsv_ms")
        road_ms = timings.get("road_ms")
        matching_ms = max(0.0, match_total - (hsv_ms or 0.0) - (road_ms or 0.0)) if mode != "none" else None
        tracking_ms = max(0.0, (elapsed - search_start) * 1000 - match_total) if tracking else None
        stats = panel = None
        diagnostic_ms = None
        if observation is not None:
            shadow = observation.shadow
            effective = (shadow.effective_coverage if self.lighting.shadow_aware else 1.0) if shadow else None
            stats = LightingStats(self.lighting.corrector.config.method, self.lighting.shadow_aware,
                                  effective, shadow.shadow_ratio if shadow else None,
                                  (search_start - start) * 1000,
                                  match_total)
            if self.lighting.debug and not self.road_debug:
                diagnostic_start = perf_counter()
                panel = diagnostic_panel(original, observation, self.matcher.detector,
                                         self.matcher.detector.label_palette,
                                         labels=debug_labels[0] if debug_labels else None)
                diagnostic_ms = (perf_counter() - diagnostic_start) * 1000
        if self.road_debug:
            diagnostic_start = perf_counter()
            labels = (debug_labels[0] if debug_labels else
                      self.matcher.detector.detect(image, include_contours=False).labels)
            road = debug_roads[0] if debug_roads else detect_roads(image, labels, confidence)
            accepted = measurement and measurement.success and (tracking is None or tracking.measurement_used)
            projection = (self.matcher.project_roads(image.shape, measurement) if accepted
                          else np.zeros(image.shape[:2], np.uint8))
            panel = road_panel(road_views(original, self.matcher.detector.label_palette, labels, road, projection))
            diagnostic_ms = (perf_counter() - diagnostic_start) * 1000
        projected_xy = projected_at = None
        if tracking is not None and tracking.xy_m is not None and frame.received_at is not None:
            # Output projection leaves the posterior at the observation epoch for the next step.
            projected_at = perf_counter()
            observed_at = frame.received_at if frame.observation_at is None else frame.observation_at
            lag = max(0.0, projected_at - observed_at)
            projected_xy = tuple(x + lag * v for x, v in zip(tracking.xy_m, tracking.velocity_mps))
        return LiveResult(
            measurement, tracking, (perf_counter() - start) * 1000, aspect, stats, panel,
            preprocessing_ms=(preprocessed - start) * 1000,
            lighting_ms=(corrected - preprocessed) * 1000 if observation is not None else None,
            hsv_ms=hsv_ms, match_mode=mode, matching_ms=matching_ms, tracking_ms=tracking_ms,
            localization_ms=(elapsed - start) * 1000,
            diagnostic_ms=diagnostic_ms, projected_xy_m=projected_xy,
            projected_timestamp=projected_at,
            road_ms=road_ms,
        )
