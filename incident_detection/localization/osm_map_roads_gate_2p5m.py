"""Regenerate a comparison map using the shared campus-map builder."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from map_tools.build_campus_map import build_campus_map


if __name__ == "__main__":
    data = REPO_ROOT / "data"
    build_campus_map(data / "gis" / "KAIST_map.osm", data / "experiments" / "incident_camera" / "osm_map")
