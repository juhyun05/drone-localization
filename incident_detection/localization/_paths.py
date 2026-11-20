"""Make the sibling LLM_detects modules importable, and name the folders this one reads from.

Dependencies go one way only: this folder imports LLM_detects modules (detector, schema, config,
dataset) for the LLM step, and nothing in LLM_detects imports this folder. Module names here must
not repeat LLM_detects names (config.py, detector.py, ...), or they would shadow each other.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
LLM_DETECTS = HERE.parent / "LLM_detects"
DATA = REPO_ROOT / "data"

if not (LLM_DETECTS / "detector.py").exists():
    raise ImportError(f"LLM_detects with detector.py is expected next to this folder: {LLM_DETECTS}")
if str(LLM_DETECTS) not in sys.path:
    sys.path.insert(0, str(LLM_DETECTS))
