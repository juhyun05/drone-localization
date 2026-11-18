"""Check that faster paths preserve label and match decisions."""

from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.illumination_correction import IlluminationConfig
from localization.image_processing import BuildingDetector, ColorConfig
from localization.lighting_pipeline import LightingProcessor
from localization.map_loader import CampusMap, MapRaster
from localization.map_matching import MapMatcher, SearchConfig
from tools.benchmark_live_pipeline import trial_orders


def reference_detect(frame, palette, config, *, rgb=False):
    """Full-frame connected components with the original HSV rule."""
    base = cv2.cvtColor(np.asarray([palette], np.uint8), cv2.COLOR_BGR2HSV)[0].astype(int)
    circular = np.abs(np.arange(180)[:, None] - base[:, 0])
    circular = np.minimum(circular, 180 - circular)
    hue_choices = np.argmin(circular, axis=1)
    hue, sat, value = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV if rgb else cv2.COLOR_BGR2HSV).transpose(2, 0, 1)
    nearest = hue_choices[hue]
    valid = ((circular[hue, nearest] <= config.hue_tolerance)
             & (abs(sat - base[nearest, 1]) <= config.saturation_tolerance)
             & (abs(value - base[nearest, 2]) <= config.value_tolerance))
    raw = np.where(valid, nearest + 1, 0).astype(np.uint8)
    labels, contours = np.zeros_like(raw), {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_size, config.morph_size))
    for color in np.unique(raw[valid]):
        mask = cv2.morphologyEx((raw == color).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, components, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = stats[:, cv2.CC_STAT_AREA] >= config.min_area_px
        keep[0] = False
        if keep.any():
            cleaned = keep[components]
            labels[cleaned & ((labels == 0) | (raw == color))] = color
            contours[int(color)], _ = cv2.findContours(cleaned.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
    return labels, contours


class PerformanceEquivalenceTest(TestCase):
    def test_physical_hue_shift_wraps_and_keeps_saturation_gate(self):
        hsv_palette = np.array([[[0, 150, 230], [177, 150, 230]]], np.uint8)
        palette = tuple(map(tuple, cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR)[0]))
        hsv = np.full((24, 32, 3), (177, 150, 230), np.uint8)
        hsv[:, 16:, 1] = 20
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        default_labels = BuildingDetector(palette).detect(frame, include_contours=False).labels
        calibrated = BuildingDetector(palette, ColorConfig(hue_shift=3))
        labels = calibrated.detect(frame, include_contours=False).labels
        np.testing.assert_array_equal(default_labels[:, :16], 2)
        np.testing.assert_array_equal(labels[:, :16], 1)
        np.testing.assert_array_equal(labels[:, 16:], 0)

    def test_printed_hue_bands_use_the_same_labels_for_camera_and_map(self):
        campus = CampusMap.load()
        matcher = MapMatcher(campus, SearchConfig(auto_scale=True),
                             ColorConfig(hue_band_width=15))
        labels = matcher.detector.label_map
        self.assertEqual(len(matcher.detector.label_palette), 12)
        self.assertEqual(labels[0], 0)
        for engine in (matcher, matcher.coarse):
            np.testing.assert_array_equal(engine.labels, labels[engine.raster.labels()])
        first, second = next((i, j) for i in range(1, len(labels))
                             for j in range(i + 1, len(labels)) if labels[i] == labels[j])
        image = np.empty((24, 32, 3), np.uint8)
        image[:, :16] = campus.palette[first - 1]
        image[:, 16:] = campus.palette[second - 1]
        actual = matcher.detector.detect(image, include_contours=False).labels
        np.testing.assert_array_equal(actual, labels[first])

    def test_segmentation_preserves_wrap_ties_borders_and_contours(self):
        palette = ((0, 0, 255), (0, 18, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0))
        rng = np.random.default_rng(74)
        image = rng.integers(0, 256, size=(71, 107, 3), dtype=np.uint8)
        image[:14, :13] = (9, 0, 240)
        image[20:32, 38:57] = (0, 18, 233)
        image[-10:, -21:] = (255, 0, 0)
        for config in (ColorConfig(), ColorConfig(90, 255, 255, 1, 1), ColorConfig(morph_size=5)):
            for rgb in (False, True):
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if rgb else image[:, ::-1]
                expected, contours = reference_detect(frame, palette, config, rgb=rgb)
                result = BuildingDetector(palette, config).detect(frame, rgb=rgb)
                np.testing.assert_array_equal(result.labels, expected)
                self.assertEqual(result.contours.keys(), contours.keys())
                for color, outlines in result.contours.items():
                    self.assertEqual(len(outlines), len(contours[color]))
                    for actual, wanted in zip(outlines, contours[color]):
                        np.testing.assert_array_equal(actual, wanted)

    def test_connected_components_uses_color_roi(self):
        image = np.zeros((240, 320, 3), np.uint8)
        image[90:115, 110:155] = (0, 0, 255)
        detector = BuildingDetector(((0, 0, 255),))
        with patch("localization.image_processing.cv2.connectedComponentsWithStats",
                   wraps=cv2.connectedComponentsWithStats) as components:
            labels = detector.detect(image, include_contours=False).labels
        self.assertEqual(np.count_nonzero(labels), 25 * 45 - 4)
        self.assertEqual(components.call_args.args[0].shape, (25, 45))

    def test_prepared_empty_paper_sample_is_not_recomputed(self):
        image = np.full((32, 40, 3), (255, 0, 0), np.uint8)
        lighting = LightingProcessor(IlluminationConfig("combined"), shadow_aware=True)
        with patch("localization.illumination_correction.sample_paper",
                   side_effect=AssertionError("already sampled")):
            with patch("localization.shadow_detection.sample_paper",
                       side_effect=AssertionError("already sampled")):
                result = lighting.apply(image)
        np.testing.assert_array_equal(result.image, image)
        np.testing.assert_array_equal(result.confidence, 1)

    def test_opencv_thread_settings_preserve_weighted_auto_scale_estimate(self):
        campus = CampusMap.load()
        camera = SyntheticCamera(MapRaster(campus, 320), CameraConfig())
        image = camera.capture(Pose(1.73, 2.02, 37))
        weights = np.linspace(.4, 1, image.shape[1], dtype=np.float32)
        confidence = np.broadcast_to(weights, image.shape[:2]).copy()
        original = cv2.getNumThreads()
        results = []
        try:
            for threads in (1, 2, 4, original):
                cv2.setNumThreads(threads)
                matcher = MapMatcher(campus, SearchConfig(auto_scale=True))
                results.append(matcher.estimate(image, (1.73, 2.02), prior_yaw=37,
                                                prior_span_m=camera.config.footprint_m,
                                                confidence_mask=confidence))
        finally:
            cv2.setNumThreads(original)
        self.assertTrue(results[0].success)
        for result in results[1:]:
            self.assertEqual(result, results[0])

    def test_benchmark_orders_repeat_and_rotate_reproducibly(self):
        periods = (0, 100, 150, 200)
        orders = list(trial_orders(periods, 3, 42))
        self.assertEqual(orders, list(trial_orders(periods, 3, 42)))
        self.assertEqual(len({order[0] for order in orders}), 3)
        self.assertTrue(all(set(order) == set(periods) for order in orders))
