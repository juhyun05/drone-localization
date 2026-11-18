"""Explicit frame transforms for physical camera input."""

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np
import yaml

from .camera_simulator import footprint_width

DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parents[1] / "config" / "camera_config.yaml"


@dataclass(frozen=True)
class PhysicalCameraConfig:
    size: tuple[int, int]
    rotation: int
    flip_horizontal: bool
    flip_vertical: bool
    calibration_path: Path | None
    brightness: float
    gains: tuple[float, float, float]
    nominal_span_m: float | None

    @classmethod
    def load(cls, path=DEFAULT_CAMERA_CONFIG):
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"invalid camera YAML: {error}") from error
        if not isinstance(data, dict):
            raise ValueError("camera YAML must be a mapping")
        size = data.get("resolution", {})
        orientation = data.get("orientation", {})
        calibration = data.get("calibration", {})
        light = data.get("color", {})
        geometry = data.get("localization", {})
        if not all(isinstance(section, dict) for section in (size, orientation, calibration, light, geometry)):
            raise ValueError("camera YAML sections must be mappings")
        dimensions = size.get("width", 320), size.get("height", 240)
        rotation = orientation.get("rotation_deg", 0)
        flip_h, flip_v = (orientation.get(key, False) for key in ("flip_horizontal", "flip_vertical"))
        enabled = calibration.get("enabled", False)
        filename = calibration.get("calibration_file")
        if (not all(type(value) is int and value > 0 for value in dimensions)
                or rotation not in (0, 90, 180, 270)):
            raise ValueError("resolution must be positive and rotation must be 0, 90, 180 or 270")
        if not all(type(value) is bool for value in (flip_h, flip_v, enabled)):
            raise ValueError("flip and calibration switches must be YAML booleans")
        if enabled and not isinstance(filename, str):
            raise ValueError("enabled calibration needs calibration_file")
        try:
            brightness = float(light.get("brightness", 1))
            gains = tuple(map(float, light.get("bgr_gains", [1, 1, 1])))
        except (TypeError, ValueError) as error:
            raise ValueError("brightness and BGR gains must be numeric") from error
        if len(gains) != 3 or not all(math.isfinite(value) and value > 0 for value in (brightness, *gains)):
            raise ValueError("brightness and three BGR gains must be positive and finite")
        span = geometry.get("nominal_span_m")
        height, fov = geometry.get("height_m"), geometry.get("hfov_deg")
        if span is None and height is not None and fov is not None:
            span = footprint_width(float(height), float(fov))
        if span is not None and not (math.isfinite(float(span)) and float(span) > 0):
            raise ValueError("nominal_span_m must be positive and finite")
        return cls(dimensions, rotation, flip_h, flip_v, path.parent / filename if enabled else None,
                   brightness, gains, None if span is None else float(span))


class CameraPreprocessor:
    def __init__(self, config: PhysicalCameraConfig):
        self.config = config
        self.maps = None
        self.map_size = None
        if config.calibration_path:
            with np.load(config.calibration_path, allow_pickle=False) as data:
                self.k, self.dist = data["camera_matrix"], data["dist_coeffs"]
                self.calibration_size = tuple(map(int, data["image_size"]))
            if (self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                    or not np.isfinite(self.dist).all() or self.dist.size not in (4, 5, 8, 12, 14)
                    or len(self.calibration_size) != 2 or min(self.calibration_size) <= 0):
                raise ValueError("malformed camera calibration file")

    def apply(self, frame):
        """Normalize uint8 BGR without cropping or stretching the field of view."""
        if frame.dtype != np.uint8 or not frame.size:
            raise ValueError("expected a nonempty uint8 camera frame")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected gray, BGR or BGRA")
        config = self.config
        if config.calibration_path:
            size = frame.shape[1], frame.shape[0]
            if size != self.calibration_size:
                raise ValueError(f"capture {size} differs from calibration {self.calibration_size}")
            if self.map_size != size:
                self.maps = cv2.initUndistortRectifyMap(
                    self.k, self.dist, None, self.k, size, cv2.CV_16SC2)
                self.map_size = size
            frame = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                     270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if config.rotation:
            frame = cv2.rotate(frame, rotations[config.rotation])
        if config.flip_horizontal or config.flip_vertical:
            axis = -1 if config.flip_horizontal and config.flip_vertical else int(config.flip_horizontal)
            frame = cv2.flip(frame, axis)
        height, width = frame.shape[:2]
        fraction = min(config.size[0] / width, config.size[1] / height)
        output = max(1, round(width * fraction)), max(1, round(height * fraction))
        if output != (width, height):
            frame = cv2.resize(frame, output, interpolation=cv2.INTER_AREA if fraction < 1 else cv2.INTER_LINEAR)
        if config.brightness != 1 or config.gains != (1, 1, 1):
            factors = config.brightness * np.asarray(config.gains, dtype=np.float32)
            frame = np.clip(np.rint(frame.astype(np.float32) * factors), 0, 255).astype(np.uint8)
        return frame
