"""Settings of the LLM incident detector. CLI flags in evaluate.py override Config.

The photos and labels live in the repository's data/photos/ directory. Map, camera and
photo-finder settings are in ../localization/settings.py."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PHOTOS = PROJECT / "data" / "photos"
OUTPUTS = PROJECT / "data" / "experiments" / "photo_classifier"

# USD per 1M tokens. Output includes reasoning tokens. Tokens the API writes to its prompt cache
# are billed at the cache_write rate instead of the input rate (input rate when none is listed).
# Checked 2026-11-16 at https://developers.openai.com/api/docs/pricing and each model's page
PRICES = {
    "gpt-6-luna": {"input": 0.10, "cached": 0.01, "cache_write_30m": 0.125, "output": 0.50},
    "gpt-6.1-terra": {"input": 1.00, "cached": 0.05, "cache_write_30m": 1.25, "cache_write_12h": 2.50,
                      "output": 5.00},
    "gpt-5-mini": {"input": 0.25, "cached": 0.025, "output": 2.00},
    "gpt-5.4-mini": {"input": 0.75, "cached": 0.075, "output": 4.50},
}


@dataclass(frozen=True)
class Config:
    # Language of free-text fields; rendered into prompts as {{ language }}
    language: str = "English"

    # Model
    model: str = "gpt-6-luna"
    reasoning_effort: str = "low"  # none | low | medium | high | xhigh | max
    image_detail: str = "auto"  # low | high | auto
    max_output_tokens: int = 4000
    timeout_s: float = 60.0
    max_retries: int = 2

    # Prompt: the model picks its own category; these are only examples
    category_examples: tuple[str, ...] = (
        "fire",
        "medical_emergency",
        "unauthorized_access",
        "illegal_parking",
        "vandalism",
        "theft",
        "flooding",
        "fight",
    )
    # Keys must match schema.Urgency values
    urgency_definitions: dict[str, str] = field(default_factory=lambda: {
        "none": "Normal scene. No action needed.",
        "low": "Minor issue. Log it and check when convenient.",
        "medium": "Needs an officer soon, but no immediate danger.",
        "high": "Immediate danger to people or property. Dispatch now.",
    })

    # Paths
    prompts_dir: Path = ROOT / "prompts"
    image_dir: Path = PHOTOS / "photo_unlabeled"
    labels_csv: Path = PHOTOS / "labels.csv"
    outputs_dir: Path = OUTPUTS
    spend_log: Path = OUTPUTS / "spend.csv"

    # Budget: whole-project prepaid credit; calls stop once logged spend reaches stop_at_usd
    total_budget_usd: float = 15.0
    stop_at_usd: float = 12.0
