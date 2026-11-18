"""Deterministic illumination perturbations for existing synthetic camera frames."""

import cv2
import numpy as np


LIGHTING_CASES = ("clear", "uniform", "local", "soft", "moving", "complex", "severe")


def simulate_lighting(frame, case, *, phase=0.5, seed=42):
    """Return a BGR frame and the synthetic shadow truth, for evaluation only."""
    if case not in LIGHTING_CASES or not 0 <= phase <= 1:
        raise ValueError("unknown lighting case or phase outside [0, 1]")
    height, width = frame.shape[:2]
    shadow = np.zeros((height, width), np.float32)
    if case == "clear":
        return frame.copy(), shadow.astype(bool)
    if case in ("local", "soft", "moving", "complex", "severe"):
        x = 0.22 + 0.56 * phase if case == "moving" else 0.51
        axes = (0.42, 0.43) if case == "severe" else (0.33, 0.40)
        cv2.ellipse(shadow, (round(width * x), round(height * 0.51)),
                    (round(width * axes[0]), round(height * axes[1])), -17, 0, 360, 1, -1)
        if case != "local":
            shadow = cv2.GaussianBlur(shadow, (0, 0), max(1, min(height, width) * 0.045))
    depth = 0.92 if case == "severe" else 0.60
    transmission = 1 - depth * shadow
    if case == "uniform":
        transmission.fill(0.43)
    cast = np.array([1.02, 1.0, 0.98], np.float32)
    values = frame.astype(np.float32) * transmission[..., None] * (1 + shadow[..., None] * (cast - 1))
    if case == "complex":
        gradient = np.linspace(0.59, 1.01, width, dtype=np.float32)
        values *= gradient[None, :, None] * np.array([0.85, 1.00, 1.11], np.float32)
        rng = np.random.default_rng(seed)
        values += rng.normal(0, 2.0, values.shape)
    image = np.rint(values).clip(0, 255).astype(np.uint8)
    return image, shadow * depth >= 0.23
