"""Create a durable stationary replay fixture for localization benchmarks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from localization.camera_interface import CameraFrame
from localization.camera_recording import CameraRecorder
from localization.camera_simulator import CameraConfig, Pose, SyntheticCamera
from localization.map_loader import CampusMap, DEFAULT_MAP, MapRaster
from tools.benchmark_environment import file_digest


def generate(output):
    output = Path(output)
    pose = Pose(1.73, 2.02, 37)
    fps, count, seed = 25, 1800, 42
    campus = CampusMap.load()
    camera = SyntheticCamera(MapRaster(campus, 320), CameraConfig(), seed=seed)
    image = camera.capture(pose)
    members = ("stationary.avi", "stationary.csv", "stationary.json", "manifest.json")
    if any((output / member).exists() for member in members):
        raise ValueError("fixture output already exists")
    output.mkdir(parents=True, exist_ok=True)
    recipe = {"input_kind": "synthetic", "scenario": "stationary", "frame_count": count,
              "timeline_fps": fps, "seed": seed, "pose_xy_yaw": [pose.x_m, pose.y_m, pose.yaw_deg],
              "map_sha256": file_digest(DEFAULT_MAP), "source_image_sha256": sha256(image.tobytes()).hexdigest(),
              "opencv": cv2.__version__, "numpy": np.__version__}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "stationary.avi"
        recorder = CameraRecorder(path, recipe)
        try:
            for index in range(count):
                recorder.write(CameraFrame(image, index / fps, index + 1, "synthetic"))
        finally:
            recorder.close()
        for member in members[:-1]:
            shutil.copyfile(path.parent / member, output / member)
    manifest = {**recipe, "files": {member: file_digest(output / member) for member in members[:-1]}}
    with (output / members[-1]).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
