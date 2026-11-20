"""Run the detector on the test photos and score is_abnormal against labels.csv.

The photos are test data only: the model sees one unlabeled photo at a time, and labels are
compared after it answers. Do not tune the prompt on these results.

Usage:
    python evaluate.py --limit 2                    # smoke test on the first 2 photos
    python evaluate.py                              # all photos with config.py settings
    python evaluate.py --model gpt-6.1-terra        # same photos, another model
    python evaluate.py --image-dir photo_drone      # another photo set named by id (1.jpg, 2.jpg, ...),
                                                    # a path or a folder name in data/photos/
    python evaluate.py --image-dir photo_gray       # the same photos in grayscale, like the new camera
    python evaluate.py --resume RUN_ID              # finish an interrupted run, retrying failed photos
"""
import argparse
import csv
import dataclasses
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath

from openai import OpenAIError
from pydantic import ValidationError

from config import PHOTOS, PROJECT, ROOT, Config
from dataset import find_photo, read_labels
from detector import BudgetExceeded, Detector, consistency_warnings, total_spent

# Settings restored from config.json on --resume, so one run never mixes settings
RUN_KEYS = ("model", "reasoning_effort", "language", "image_detail", "max_output_tokens", "image_dir")


def find_image_dir(path: Path, *bases: Path) -> Path:
    """A photo folder: `path` itself if absolute, else under the first of `bases` that has it, else the
    folder of the same name in data/photos/ (handles runs saved before the move)."""
    tries = [path] if path.is_absolute() else [base / path for base in bases]
    tries.append(PHOTOS / PureWindowsPath(path).name)
    return next((p.resolve() for p in tries if p.is_dir()), tries[0])


def label(abnormal: bool) -> str:
    return "abnormal" if abnormal else "normal"


def load_records(path: Path) -> dict[str, dict]:
    """Last record per photo wins, so a retried photo replaces its failed attempt."""
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[record["image_id"]] = record
    return records


def summarize(cfg: Config, run_id: str, records: dict[str, dict]) -> str:
    by_id = lambda r: int(r["image_id"])
    ok = sorted((r for r in records.values() if "error" not in r), key=by_id)
    failed = sorted((r for r in records.values() if "error" in r), key=by_id)
    det = lambda r: r["report"]["detection"]
    tp = sum(r["truth_abnormal"] and det(r)["is_abnormal"] for r in ok)
    fn = sum(r["truth_abnormal"] and not det(r)["is_abnormal"] for r in ok)
    fp = sum(not r["truth_abnormal"] and det(r)["is_abnormal"] for r in ok)
    tn = len(ok) - tp - fn - fp
    caught = sum(r["truth_abnormal"] and (det(r)["is_abnormal"] or det(r)["needs_human_review"]) for r in ok)
    ratio = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "n/a"
    image_dir = Path(cfg.image_dir)
    shown_dir = image_dir.relative_to(PROJECT) if image_dir.is_relative_to(PROJECT) else image_dir

    calls = []
    if cfg.spend_log.exists():
        with open(cfg.spend_log, newline="") as f:
            calls = [c for c in csv.DictReader(f) if c["run_id"] == run_id]
    avg = lambda key: sum(int(c[key]) for c in calls) / len(calls) if calls else 0
    run_cost = sum(float(c["cost_usd"]) for c in calls)

    lines = [
        f"# Run {run_id}",
        "",
        f"- Model: {cfg.model}, reasoning effort: {cfg.reasoning_effort}, image detail: {cfg.image_detail}, "
        f"language: {cfg.language}",
        f"- Photos: {shown_dir}, scored {len(ok)}, failed {len(failed)}",
        "",
        "## is_abnormal vs labels.csv",
        "",
        "| | pred abnormal | pred normal |",
        "|---|---|---|",
        f"| **truth abnormal** | {tp} (TP) | {fn} (FN) |",
        f"| **truth normal** | {fp} (FP) | {tn} (TN) |",
        "",
        f"- Accuracy: {ratio(tp + tn, len(ok))}",
        f"- Abnormal recall: {ratio(tp, tp + fn)}",
        f"- Abnormal precision: {ratio(tp, tp + fp)}",
        f"- False alarms on normal photos: {ratio(fp, fp + tn)}",
        f"- Abnormal photos caught by is_abnormal or needs_human_review: {ratio(caught, tp + fn)}",
        "",
        "## Cost",
        "",
        f"- This run: ${run_cost:.4f} over {len(calls)} calls (avg {avg('input_tokens'):.0f} input / "
        f"{avg('output_tokens'):.0f} output tokens per call, {avg('reasoning_tokens'):.0f} of them reasoning)",
        f"- All logged spend: ${total_spent(cfg.spend_log):.4f} of ${cfg.total_budget_usd:.2f} "
        f"(calls stop at ${cfg.stop_at_usd:.2f})",
        "",
        "## Per photo",
        "",
        "| id | photo | truth | pred | category | urgency | quality | review | warnings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ok:
        d = det(r)
        miss = "" if d["is_abnormal"] == r["truth_abnormal"] else " ✗"
        lines.append(f"| {r['image_id']} | {r['labeled_file']} | {label(r['truth_abnormal'])} | "
                     f"{label(d['is_abnormal'])}{miss} | {d['category']} | {d['urgency']} | {d['image_quality']} | "
                     f"{'yes' if d['needs_human_review'] else 'no'} | {'; '.join(r['warnings']) or '-'} |")
    for r in failed:
        error = r["error"].replace("|", "/")[:200]
        lines.append(f"| {r['image_id']} | {r['labeled_file']} | {label(r['truth_abnormal'])} | "
                     f"ERROR | {error} | | | | |")
    lines += ["", "## Descriptions", ""]
    for r in ok:
        lines += [f"**{r['image_id']}** ({r['labeled_file']}): {det(r)['description']}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--effort", dest="reasoning_effort")
    parser.add_argument("--language")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--limit", type=int, help="only the first N photos by id")
    parser.add_argument("--resume", metavar="RUN_ID")
    args = parser.parse_args()
    overrides = {k: v for k in ("model", "reasoning_effort", "language", "image_dir")
                 if (v := getattr(args, k)) is not None}

    if args.resume:
        if overrides:
            parser.error("--resume reuses the run's own settings; drop the other setting flags")
        run_dir = Config().outputs_dir / args.resume
        saved = json.loads((run_dir / "config.json").read_text())
        restored = {k: saved[k] for k in RUN_KEYS}
        # Newer runs save paths relative to the project folder, older ones relative to LLM_detects
        restored["image_dir"] = find_image_dir(Path(restored["image_dir"]), PROJECT, ROOT)
        cfg = dataclasses.replace(Config(), **restored)
    else:
        if "image_dir" in overrides:
            overrides["image_dir"] = find_image_dir(overrides["image_dir"], Path.cwd())
        cfg = dataclasses.replace(Config(), **overrides)
        run_dir = cfg.outputs_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model}"
        run_dir.mkdir(parents=True)

    detector = Detector(cfg)
    if args.resume:
        if (run_dir / "system_prompt.txt").read_text(encoding="utf-8") != detector.system_prompt:
            raise SystemExit("Prompts changed since this run started; start a new run instead of resuming")
    else:
        # Paths inside the project are saved relative to it, so a run can be resumed after a folder move
        saved = {k: v.relative_to(PROJECT).as_posix() if isinstance(v, Path) and v.is_relative_to(PROJECT) else v
                 for k, v in dataclasses.asdict(cfg).items()}
        (run_dir / "config.json").write_text(json.dumps(saved, indent=2, default=str))
        (run_dir / "system_prompt.txt").write_text(detector.system_prompt, encoding="utf-8")

    rows = read_labels(cfg.labels_csv)[:args.limit]
    results_path = run_dir / "results.jsonl"
    done = {i for i, r in load_records(results_path).items() if "error" not in r}
    todo = [r for r in rows if r["id"] not in done]
    print(f"Run {run_dir.name}: {cfg.model}, effort={cfg.reasoning_effort}, {len(todo)} photos to go")

    with open(results_path, "a", encoding="utf-8") as out:
        for n, row in enumerate(todo, 1):
            # The label stays here for scoring; the detector only gets the unlabeled photo
            record = {"image_id": row["id"], "labeled_file": row["labeled_file"],
                      "truth_abnormal": row["label"] == "abnormal"}
            try:
                report, usage = detector.detect(find_photo(cfg.image_dir, row["id"]), row["id"], run_dir.name)
            except BudgetExceeded as e:
                print(f"Stopped: {e}")
                break
            except (OpenAIError, ValidationError, RuntimeError, OSError) as e:
                record["error"] = f"{type(e).__name__}: {e}"
                print(f"[{n}/{len(todo)}] id={row['id']} ERROR {record['error']}")
            else:
                d = report.detection
                record.update(report=report.model_dump(mode="json"), usage=usage.model_dump(),
                              warnings=consistency_warnings(d))
                print(f"[{n}/{len(todo)}] id={row['id']} truth={label(record['truth_abnormal'])} "
                      f"pred={label(d.is_abnormal)} category={d.category} urgency={d.urgency.value} "
                      f"review={d.needs_human_review} ${usage.cost_usd:.5f}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(cfg, run_dir.name, load_records(results_path))
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
