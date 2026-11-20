"""The test photos: find one by id, read labels.csv, and make the grayscale copies.

The 18 photos are test data only: no training, no few-shot examples, no prompt tuning on their results.

Usage:
    python dataset.py gray    # data/photos/photo_unlabeled/ -> data/photos/photo_gray/ (as the grayscale camera sees them)
"""
import csv
import sys
from pathlib import Path

from config import Config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def find_photo(image_dir: Path, image_id: str) -> Path:
    """The single photo named <image_id>.<png|jpg|jpeg|webp> in image_dir."""
    matches = [p for p in Path(image_dir).glob(f"{image_id}.*") if p.suffix.lower() in IMAGE_SUFFIXES]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one photo named {image_id}.* in {image_dir}, found {len(matches)}")
    return matches[0]


def read_labels(labels_csv: Path) -> list[dict]:
    """Rows of labels.csv (id, unlabeled_file, labeled_file, group, label), sorted by id."""
    with open(labels_csv, newline="", encoding="utf-8") as f:
        return sorted(csv.DictReader(f), key=lambda r: int(r["id"]))


def make_gray(cfg: Config = Config(), out_dir: Path | None = None) -> Path:
    """Grayscale copy of every test photo, same ids, as PNG."""
    import cv2  # only needed here
    from image_io import read_image, write_image
    out_dir = out_dir or cfg.image_dir.parent / "photo_gray"
    out_dir.mkdir(exist_ok=True)
    for row in read_labels(cfg.labels_csv):
        gray = read_image(find_photo(cfg.image_dir, row["id"]), cv2.IMREAD_GRAYSCALE)
        write_image(out_dir / f"{row['id']}.png", gray)
    return out_dir


if __name__ == "__main__":
    if sys.argv[1:] != ["gray"]:
        sys.exit(__doc__)
    print(f"wrote {make_gray()}")
