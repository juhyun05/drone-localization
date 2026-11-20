"""Render the prompts, send one photo to the model, and wrap the answer in a Report.

Usage:
    python detector.py PHOTO    # prints the Report JSON the app would receive
"""
import base64
import csv
import mimetypes
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import OpenAI

from config import PRICES, ROOT, Config
from schema import Detection, ImageQuality, Location, Report, Urgency, Usage

SPEND_FIELDS = ["timestamp", "run_id", "model", "image_id", "input_tokens", "cached_tokens",
                "cache_write_tokens", "output_tokens", "reasoning_tokens", "cost_usd"]

# Structured Outputs: the API only returns JSON that matches this schema
DETECTION_FORMAT = {
    "type": "json_schema",
    "name": "detection",
    "schema": Detection.model_json_schema(),
    "strict": True,
}


class BudgetExceeded(RuntimeError):
    pass


def total_spent(spend_log: Path) -> float:
    if not spend_log.exists():
        return 0.0
    with open(spend_log, newline="") as f:
        return sum(float(row["cost_usd"]) for row in csv.DictReader(f))


def consistency_warnings(d: Detection) -> list[str]:
    """Flag answers that break the prompt rules. Values are kept as the model gave them."""
    warnings = []
    if d.is_abnormal == (d.urgency == Urgency.NONE):
        warnings.append(f"is_abnormal={d.is_abnormal} but urgency={d.urgency.value}")
    if d.is_abnormal == (d.category == "normal"):
        warnings.append(f"is_abnormal={d.is_abnormal} but category={d.category}")
    if d.image_quality == ImageQuality.UNUSABLE and not d.needs_human_review:
        warnings.append("image_quality=unusable but needs_human_review=false")
    return warnings


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class Detector:
    def __init__(self, cfg: Config):
        if cfg.model not in PRICES:
            raise ValueError(f"No price for {cfg.model!r}; add it to PRICES in config.py so spend is tracked")
        if set(cfg.urgency_definitions) != {u.value for u in Urgency}:
            raise ValueError("urgency_definitions keys must match schema.Urgency values")
        load_dotenv(ROOT / ".env")
        self.cfg = cfg
        self.client = OpenAI(timeout=cfg.timeout_s, max_retries=cfg.max_retries)
        env = Environment(loader=FileSystemLoader(cfg.prompts_dir), undefined=StrictUndefined,
                          trim_blocks=True, lstrip_blocks=True)
        self.system_prompt = env.get_template("system.j2").render(
            language=cfg.language,
            category_examples=cfg.category_examples,
            urgency_definitions=cfg.urgency_definitions,
        )
        self.user_template = env.get_template("user.j2")

    def detect(self, image_path: Path, image_id: str, run_id: str = "adhoc",
               captured_at: datetime | None = None,
               location: Location | None = None) -> tuple[Report, Usage]:
        spent = total_spent(self.cfg.spend_log)
        if spent >= self.cfg.stop_at_usd:
            raise BudgetExceeded(f"logged spend ${spent:.2f} reached stop_at_usd ${self.cfg.stop_at_usd:.2f}")

        user_text = self.user_template.render(image_id=image_id, captured_at=captured_at, location=location)
        response = self.client.responses.create(
            model=self.cfg.model,
            instructions=self.system_prompt,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": _data_url(image_path), "detail": self.cfg.image_detail},
            ]}],
            text={"format": DETECTION_FORMAT},
            reasoning={"effort": self.cfg.reasoning_effort},
            max_output_tokens=self.cfg.max_output_tokens,
        )
        # Log before validating so billed-but-failed calls still count toward the budget
        usage = self._usage(response.usage)
        self._log_spend(run_id, image_id, usage)
        if response.status != "completed":
            raise RuntimeError(f"response {response.status}: {response.incomplete_details}")
        detection = Detection.model_validate_json(response.output_text)

        report = Report(image_id=image_id, image_path=str(image_path), captured_at=captured_at,
                        location=location, model=self.cfg.model, detection=detection)
        return report, usage

    def _usage(self, u) -> Usage:
        p = PRICES[self.cfg.model]
        details = u.input_tokens_details
        cached = details.cached_tokens
        writes = getattr(details, "cache_write_tokens", 0) or 0
        writes_12h = getattr(getattr(details, "cache_write_tokens_details", None), "ttl_12h_tokens", 0) or 0
        rate_30m = p.get("cache_write_30m", p["input"])
        cost = ((u.input_tokens - cached - writes) * p["input"] + cached * p["cached"]
                + (writes - writes_12h) * rate_30m + writes_12h * p.get("cache_write_12h", rate_30m)
                + u.output_tokens * p["output"]) / 1e6
        return Usage(input_tokens=u.input_tokens, cached_tokens=cached, cache_write_tokens=writes,
                     output_tokens=u.output_tokens, reasoning_tokens=u.output_tokens_details.reasoning_tokens,
                     cost_usd=round(cost, 8))

    def _log_spend(self, run_id: str, image_id: str, usage: Usage) -> None:
        log = self.cfg.spend_log
        log.parent.mkdir(parents=True, exist_ok=True)
        is_new = not log.exists()
        with open(log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SPEND_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
                             "model": self.cfg.model, "image_id": image_id, **usage.model_dump()})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    photo = Path(sys.argv[1])
    report, usage = Detector(Config()).detect(photo, image_id=photo.stem)
    print(report.model_dump_json(indent=2))
    print(f"cost ${usage.cost_usd:.5f}", file=sys.stderr)
