"""Alternate verified baseline and current stages on the same replay frame."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from zipfile import ZipFile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from localization import camera_simulator, illumination_correction, lighting_pipeline, map_loader, map_matching
from localization.camera_interface import VideoFileSource
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from tools.benchmark_environment import benchmark_environment


def load_reference(archive, fingerprints):
    package = ModuleType("localization_reference")
    sys.modules[package.__name__] = package
    sys.modules[f"{package.__name__}.camera_simulator"] = camera_simulator
    sys.modules[f"{package.__name__}.map_loader"] = map_loader
    modules = {}
    with ZipFile(archive) as sources:
        for stem in ("image_processing", "illumination_correction", "shadow_detection",
                     "lighting_pipeline", "map_matching"):
            member = f"localization/{stem}.py"
            body = sources.read(member)
            if sha256(body).hexdigest() != fingerprints[member]:
                raise ValueError(f"reference source hash mismatch: {member}")
            module = ModuleType(f"{package.__name__}.{stem}")
            module.__package__ = package.__name__
            sys.modules[module.__name__] = module
            exec(compile(body, f"{archive}/{member}", "exec"), vars(module))
            modules[stem] = module
    return modules


def stages(matcher, lighting, image, prior):
    observation = lighting.apply(image)
    weights = np.broadcast_to(np.linspace(.35, 1, image.shape[1], dtype=np.float32),
                              image.shape[:2]).copy()

    def segment():
        matcher.detector.detect(observation.image, include_contours=False)

    def match(confidence):
        timings = {}
        start = perf_counter()
        matcher.estimate(observation.image, **prior, timings=timings, confidence_mask=confidence)
        return (perf_counter() - start) * 1000 - timings["hsv_ms"]

    def time_call(operation):
        start = perf_counter()
        operation()
        return (perf_counter() - start) * 1000

    return {"hsv": lambda: time_call(segment),
            "lighting": lambda: time_call(lambda: lighting.apply(image)),
            "local_matching": lambda: match(observation.confidence),
            "weighted_local_matching": lambda: match(weights)}


def compare(args):
    fingerprints = json.loads(args.baseline_profile.read_text(encoding="utf-8"))["environment"]["code_sha256"]
    old = load_reference(args.baseline_sources, fingerprints)
    campus = map_loader.CampusMap.load()
    source = VideoFileSource(args.video)
    try:
        frame = source.read()
    finally:
        source.close()
    if frame is None:
        raise ValueError("video has no decodable frame")
    image = CameraPreprocessor(PhysicalCameraConfig.load()).apply(frame.image)
    reference_matcher = old["map_matching"].MapMatcher(campus, old["map_matching"].SearchConfig(auto_scale=True))
    current_matcher = map_matching.MapMatcher(campus, map_matching.SearchConfig(auto_scale=True))
    current_lighting = lighting_pipeline.LightingProcessor(
        illumination_correction.IlluminationConfig("combined"), shadow_aware=True)
    before_lighting = old["lighting_pipeline"].LightingProcessor(
        old["illumination_correction"].IlluminationConfig("combined"), shadow_aware=True)
    observation = current_lighting.apply(image)
    seed = current_matcher.estimate(observation.image, confidence_mask=observation.confidence)
    if not seed.success:
        raise ValueError("stationary frame did not provide a valid search prior")
    prior = dict(prior_xy=(seed.x_m, seed.y_m), radius_m=.13, prior_yaw=seed.yaw_deg,
                 yaw_radius_deg=25, prior_span_m=seed.span_m)
    baseline = stages(reference_matcher, before_lighting, image, prior)
    optimized = stages(current_matcher, current_lighting, image, prior)
    results = {}
    for name, before in baseline.items():
        after = optimized[name]
        before()
        after()
        pairs = []
        for index in range(args.count):
            if index % 2:
                optimized_ms, baseline_ms = after(), before()
            else:
                baseline_ms, optimized_ms = before(), after()
            pairs.append((baseline_ms, optimized_ms))
        base, current = np.asarray(pairs).T
        results[name] = dict(
            before_mean_ms=float(base.mean()), after_mean_ms=float(current.mean()),
            before_p95_ms=float(np.percentile(base, 95)), after_p95_ms=float(np.percentile(current, 95)),
            mean_reduction_ratio=float(1 - current.mean() / base.mean()), pairs_ms=pairs,
        )
    output = dict(environment=benchmark_environment("video_replay", args.video), prior=prior,
                  baseline_sources=str(args.baseline_sources), stage_timings_ms=results)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    for name, result in results.items():
        print(f"{name}: {result['before_mean_ms']:.2f} -> {result['after_mean_ms']:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline-sources", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--opencv-threads", type=int, default=1)
    args = parser.parse_args()
    if args.count < 1 or args.opencv_threads < 1:
        parser.error("sample and thread counts must be positive")
    cv2.setNumThreads(args.opencv_threads)
    compare(args)
