# LLM_detects: incident detection

Takes one photo, asks an LLM whether the scene is normal or abnormal, and returns a structured verdict wrapped in a `Report` for the campus police app.

This folder does not depend on the rest of the system. `../localization/pipeline.py` calls it with an upright photo crop and the photo's location. The test photos and labels are in `../../data/photos/`.

## Design

| Part | Design |
|---|---|
| Input | One image file, plus optional context (image id, capture time, `Location`) rendered into the user prompt |
| Prompts | Jinja2 templates in `prompts/`. The system prompt gets the answer language, example categories and urgency definitions from `Config`. |
| Model call | OpenAI Responses API with Structured Outputs: the answer must match the `Detection` JSON schema exactly. The default is `gpt-6-luna` with reasoning effort `low`, 5–10 s and about $0.0003 per photo. |
| Output order | `Detection` lists `observations` before the verdict, so the model describes the scene first and then decides. |
| Report | Code fills every field except `detection`: image id and path, capture time, location, model name. |
| Cost control | Every call's tokens and cost are logged to `../../data/experiments/photo_classifier/spend.csv` before the answer is validated, so failed calls count too. `Detector` refuses new calls once logged spend reaches `stop_at_usd` ($12 of the $15 prepaid credit). |

### `Detection` (what the LLM returns)

| Field | Meaning |
|---|---|
| `observations` | What is visible |
| `image_quality` | `good`, `degraded` or `unusable` |
| `is_abnormal` | Best guess, even when uncertain |
| `category` | Model-chosen snake_case label, `normal` when nothing is wrong |
| `urgency` | `none`, `low`, `medium` or `high` |
| `needs_human_review` | Poor image or ambiguous scene |
| `description` | Short text for the officer, in `Config.language` |

### Interface

```python
from config import Config
from detector import Detector
from schema import Location

detector = Detector(Config())
report, usage = detector.detect(crop_path, image_id="frame_03_photo1", run_id="pipeline",
                                captured_at=None,
                                location=Location(name="slot S2, ...", latitude=36.37, longitude=127.36))
report.detection.is_abnormal, report.detection.category, report.detection.urgency
usage.cost_usd
```

## Files

| File / folder | Contents |
|---|---|
| `config.py` | `Config`: model, prompt values, paths, budget; `PRICES` per model |
| `schema.py` | `Detection` (LLM output), `Location`, `Report` (bundle for the app), `Usage` |
| `prompts/system.j2`, `prompts/user.j2` | Prompt templates |
| `detector.py` | Renders the prompts, calls the model, logs spend, returns `(Report, Usage)`. `python detector.py PHOTO` runs one photo. |
| `evaluate.py` | Runs the detector on all test photos and scores `is_abnormal` against `../../data/photos/labels.csv` |
| `dataset.py` | Finds a test photo by id, reads `labels.csv`, makes grayscale copies |
| `.env` | `OPENAI_API_KEY` (never committed) |
| `../../data/experiments/photo_classifier/` | One folder per evaluation run (`config.json`, `system_prompt.txt`, `results.jsonl`, `summary.md`) and `spend.csv` |

## Evaluation

Run the commands below from `incident_detection/LLM_detects/`.

The 18 photos in `../../data/photos/` (groups: A fire, B restricted area, C road and parking, D person down) are **test data only**. They are never used for training or few-shot examples, and the prompt is not tuned on their results. The model sees one unlabeled photo at a time; labels are compared only after it answers.

```bash
python detector.py ../../data/photos/photo_unlabeled/3.png   # one photo → Report JSON
python evaluate.py                                  # all 18 photos (about $0.006)
python evaluate.py --image-dir photo_gray           # another photo set: a folder name in ../../data/photos/ or a path
python evaluate.py --resume RUN_ID                  # finish an interrupted run with its own settings
python dataset.py gray                              # ../../data/photos/photo_unlabeled/ → ../../data/photos/photo_gray/
```

Each run saves its settings and prompt, so `--resume` never mixes settings and refuses to continue if the prompts have changed.

Results with `gpt-6-luna`:

| Input | Correct `is_abnormal` |
|---|---|
| Color originals | 18/18 |
| Grayscale copies (two runs) | 17/18 both times |
| Grayscale at drone-crop size (short side 120 px, with blur, noise and JPEG) | 16/18 |

The [120 px run](../../data/experiments/photo_classifier/20261119-150043_gpt-6-luna/summary.md) includes the 18 saved verdicts and scores. Its original `photo_cam120` images were generated in a temporary folder on the recording machine. Re-scoring uses the saved records. To repeat the model calls, supply the same image set with `--image-dir PATH`.

- Every miss on grayscale input was an illegal-parking photo (group C).
- No normal photo was reported as abnormal.
