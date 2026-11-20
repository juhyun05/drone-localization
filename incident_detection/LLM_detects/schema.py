"""Data models: what the LLM returns (Detection) and what the app receives (Report)."""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


class Urgency(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImageQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"


# Filled by the LLM through Structured Outputs. The model writes fields in this order, so
# observations come before the verdict. (A comment, not a docstring: docstrings end up in the schema.)
class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict mode requires additionalProperties: false

    observations: str
    image_quality: ImageQuality
    is_abnormal: bool  # best guess, even when uncertain
    category: str  # model-chosen snake_case label, "normal" when nothing is wrong
    urgency: Urgency
    needs_human_review: bool  # uncertainty flag: poor image or ambiguous scene
    description: str


class Location(BaseModel):
    """Where the photo is. The caller fills it: localization/pipeline.py puts the photo slot, building and
    map coordinates in `name` and the real campus position (from the drone pose) in latitude/longitude."""
    name: str
    latitude: float | None = None
    longitude: float | None = None


class Report(BaseModel):
    """Bundle delivered to the campus police app. Everything except `detection` is set by code,
    never by the LLM."""
    image_id: str
    image_path: str
    captured_at: datetime | None = None
    location: Location | None = None  # None when the position is unknown (no drone pose)
    model: str
    detection: Detection


class Usage(BaseModel):
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
