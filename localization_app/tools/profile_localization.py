"""Profile stationary replay stages and save same-frame accuracy evidence."""

import argparse
from dataclasses import asdict
import cProfile
import json
import math
from pathlib import Path
import pstats
import sys
from time import perf_counter
import tracemalloc

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from benchmark_shadow import synthetic_samples
from localization.camera_interface import VideoFileSource
from localization.camera_preprocessing import CameraPreprocessor, PhysicalCameraConfig
from localization.illumination_correction import IlluminationConfig
from localization.lighting_pipeline import LightingProcessor
from localization.map_loader import CampusMap
from localization.map_matching import MapMatcher, SearchConfig, angle_error
from tools.benchmark_environment import benchmark_environment, process_resources


class CountedMatcher(MapMatcher):
    def __init__(self, *args):
        super().__init__(*args)
        self.candidates = dict(transforms=0, score_grids=0, xy_candidates=0)
        if self.coarse is not None:
            self.coarse._score_grids = self._count(self.coarse._score_grids)

    def _count(self, score_grids):
        def count(observation, spans, headings, *args, **options):
            self.candidates["transforms"] += len(spans) * len(headings)
            for grid in score_grids(observation, spans, headings, *args, **options):
                self.candidates["score_grids"] += 1
                self.candidates["xy_candidates"] += int(np.count_nonzero(grid.scores >= 0))
                yield grid
        return count

    def _score_grids(self, *args, **kwargs):
        yield from self._count(super()._score_grids)(*args, **kwargs)


def measured(operation, count):
    samples = []
    for _ in range(count):
        start = perf_counter()
        operation()
        samples.append((perf_counter() - start) * 1000)
    return dict(mean=float(np.mean(samples)), p95=float(np.percentile(samples, 95)))


def quality(campus, matcher, processor):
    rows, labels = [], {}
    for index, sample in enumerate(synthetic_samples(campus)):
        observation = processor.apply(sample.frame)
        detected = matcher.detector.detect(observation.image, include_contours=False).labels
        labels[f"frame_{index:02}"] = detected
        truth = sample.labels != 0
        foreground = detected != 0
        estimate = matcher.estimate(observation.image, confidence_mask=observation.confidence)
        error = math.dist((estimate.x_m, estimate.y_m), (sample.pose.x_m, sample.pose.y_m)) if estimate.success else None
        yaw = angle_error(estimate.yaw_deg, sample.pose.yaw_deg) if estimate.success else None
        rows.append(dict(view=sample.view, case=sample.case, **asdict(estimate),
                         effective_coverage=observation.shadow.effective_coverage,
                         building_accuracy=float((detected[truth] == sample.labels[truth]).mean()),
                         mask_iou=float((foreground & truth).sum() / (foreground | truth).sum()),
                         position_error_m=error, yaw_error_deg=yaw,
                         wrong_accepted=None if error is None else error > .05 or yaw > 15))
    return rows, labels


def profile(video, output, *, count, include_quality):
    output.mkdir(parents=True, exist_ok=True)
    campus = CampusMap.load()
    search = SearchConfig(auto_scale=True)
    matcher = MapMatcher(campus, search)
    processor = LightingProcessor(IlluminationConfig("combined"), shadow_aware=True)
    source = VideoFileSource(video)
    try:
        frame = source.read()
    finally:
        source.close()
    if frame is None:
        raise ValueError("video has no decodable frame")
    image = CameraPreprocessor(PhysicalCameraConfig.load()).apply(frame.image)
    observation = processor.apply(image)
    global_args = dict(confidence_mask=observation.confidence)
    seed = matcher.estimate(observation.image, **global_args)
    if not seed.success:
        raise ValueError(f"stationary profile failed global localization: {seed.reason}")
    local_args = dict(global_args, prior_xy=(seed.x_m, seed.y_m), radius_m=.13,
                      prior_yaw=seed.yaw_deg, yaw_radius_deg=25, prior_span_m=seed.span_m)

    def local_match():
        timings = {}
        start = perf_counter()
        matcher.estimate(observation.image, timings=timings, **local_args)
        return (perf_counter() - start) * 1000 - timings["hsv_ms"]

    def segment():
        return matcher.detector.detect(observation.image, include_contours=False).labels

    segment()
    local_match()
    profiled = dict(hsv=measured(segment, count), lighting=measured(lambda: processor.apply(image), count))
    local_samples = [local_match() for _ in range(count)]
    profiled["local_matching"] = dict(mean=float(np.mean(local_samples)),
                                     p95=float(np.percentile(local_samples, 95)))
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        processor.apply(image)
        local_match()
    profiler.disable()
    with (output / "cprofile.txt").open("x", encoding="utf-8") as stream:
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(45)
    tracemalloc.start()
    try:
        segment()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    candidate_counts = {}
    for name, options in (("global", global_args), ("local", local_args)):
        counted = CountedMatcher(campus, search)
        counted.estimate(observation.image, **options)
        candidate_counts[name] = counted.candidates
    report = dict(environment=benchmark_environment("video_replay", video), samples=count,
                  stages_ms=profiled, hsv_traced_allocation_peak_bytes=peak,
                  process_resources=process_resources(), candidate_counts=candidate_counts,
                  confidence_weighted=bool(observation.confidence.min() < 1))
    if include_quality:
        report["quality"], labels = quality(campus, matcher, processor)
        with (output / "masks.npz").open("xb") as stream:
            np.savez_compressed(stream, **labels)
    with (output / "profile.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(profiled, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opencv-threads", type=int)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or (args.opencv_threads is not None and args.opencv_threads < 1):
        parser.error("sample and thread counts must be positive")
    if args.opencv_threads is not None:
        cv2.setNumThreads(args.opencv_threads)
    profile(args.video, args.output, count=args.count, include_quality=args.quality)
