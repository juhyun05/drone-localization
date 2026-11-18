"""Check print dimensions, vector geometry, and appearance against the source map."""

import argparse
from collections import Counter
from decimal import Decimal
from math import ceil
from pathlib import Path
import re

from matplotlib.colors import to_rgb
from matplotlib.path import Path as VectorPath
import numpy as np
from PIL import Image
from pypdf import PdfReader

from create_print_map import (
    DEFAULT_INPUT, DEFAULT_OUTPUT, EPS_NAME, PDF_NAME, PNG_NAME, PT_PER_MM,
    REFERENCE_IN, SIZE_IN, SIZE_MM, SIZE_PT, SOURCE_DIR, load_map, map_features,
)

POINT_TOL = 0.00001
REFERENCE_PNG = SOURCE_DIR / "KAIST_model_reference_2p5m.png"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_box(text, name):
    match = re.search(rf"^%%{name}: (.+)$", text, re.MULTILINE)
    require(match is not None, f"missing EPS {name}")
    values = tuple(Decimal(value) for value in match[1].split())
    require(len(values) == 4, f"invalid EPS {name}")
    return values


def physical_extent(box):
    x0, y0, x1, y1 = box
    return (x1 - x0) / PT_PER_MM, (y1 - y0) / PT_PER_MM


def verify_eps(path, features):
    text = path.read_text(encoding="ascii")
    require(text.startswith("%!PS-Adobe-3.0 EPSF-3.0\n"), "missing EPS header")
    require("%%Pages: 1\n" in text and text.splitlines().count("showpage") == 1,
            "expected one unsegmented EPS page")
    bounding_box = read_box(text, "BoundingBox")
    hires_box = read_box(text, "HiResBoundingBox")
    require(bounding_box == (0, 0, ceil(SIZE_PT), ceil(SIZE_PT)), "wrong integer BoundingBox")
    require(hires_box[:2] == (0, 0), "EPS origin must be (0,0)")
    hires_mm = physical_extent(hires_box)
    require(all(abs(length - SIZE_MM) < Decimal("0.000001") for length in hires_mm),
            "HiResBoundingBox must be 2500 by 2500 mm")
    require(not re.search(r"\b(image|imagemask|colorimage|setpagedevice|scale|translate)\b", text),
            "EPS contains raster, device, or page-scaling operators")

    chunks = re.split(r"^%%MapFeature: (\S+) (\S+)\n", text, flags=re.MULTILINE)
    page = chunks[0]
    edge = str(hires_box[2])
    require(f"newpath 0 0 moveto {edge} 0 lineto {edge} {edge} lineto 0 {edge} lineto closepath clip newpath"
            in page, "missing exact page clip")
    require(f"1 1 1 setrgbcolor 0 0 {edge} {edge} rectfill" in page,
            "missing full-size white background")

    records = list(zip(chunks[1::3], chunks[2::3], chunks[3::3]))
    counts = Counter(kind for kind, _, _ in records)
    require(counts == {"building": 60, "road": 46, "road-outline": 46, "gate": 1},
            f"incorrect feature counts: {dict(counts)}")
    for (kind, feature_id, body), feature in zip(records, features, strict=True):
        require((kind, feature_id) == (feature.kind, feature.id), "incorrect feature ID or draw order")
        verify_path(body, feature)

    integer_mm = physical_extent(bounding_box)
    print(f"EPS BoundingBox: {integer_mm[0]:.6f} × {integer_mm[1]:.6f} mm (outward integer rounding)")
    print(f"EPS HiResBoundingBox: {hires_mm[0]:.6f} × {hires_mm[1]:.6f} mm")
    print("EPS paths: 60 filled buildings, 46 road centerlines and outlines, 1 gate")
    print("All path coordinates agree with the JSON-derived geometry at 1 m = 1000 mm.")
    print(f"EPS file size: {path.stat().st_size:,} bytes")


def verify_path(body, feature):
    ops = {"moveto": VectorPath.MOVETO, "lineto": VectorPath.LINETO,
           "curveto": VectorPath.CURVE4, "closepath": VectorPath.CLOSEPOLY}
    actual = []
    for line in body.splitlines():
        tokens = line.split()
        if tokens and tokens[-1] in ops:
            actual.append((ops[tokens[-1]], [float(value) for value in tokens[:-1]]))
    expected = list(feature.path.iter_segments(simplify=False, curves=True))
    require(len(actual) == len(expected), f"wrong vertex count for {feature.id}")
    for (code, coords), (vertices, expected_code) in zip(actual, expected):
        require(code == expected_code, f"wrong path operator for {feature.id}")
        if code != VectorPath.CLOSEPOLY:
            # Independently convert point coordinates from the emitted EPS back to model meters.
            np.testing.assert_allclose(np.asarray(coords) * 25.4 / 72000, vertices,
                                       rtol=0, atol=1e-10, err_msg=feature.id)

    colors = re.findall(r"^([0-9.]+) ([0-9.]+) ([0-9.]+) setrgbcolor$", body, re.MULTILINE)
    expected_colors = [to_rgb(color) for color in (feature.fill, feature.stroke) if color]
    np.testing.assert_allclose(np.asarray(colors, dtype=float), expected_colors, rtol=0, atol=1e-9,
                               err_msg=f"wrong fill/stroke color for {feature.id}")
    commands = body.splitlines()
    require(commands.count("fill") == int(bool(feature.fill)) and commands.count("stroke") == 1,
            f"wrong paint commands for {feature.id}")
    widths = re.findall(r"^([0-9.]+) setlinewidth$", body, re.MULTILINE)
    require(len(widths) == 1 and abs(float(widths[0]) - feature.width_ref_pt * SIZE_IN / REFERENCE_IN)
            < POINT_TOL, f"wrong stroke width for {feature.id}")


def verify_pdf(path):
    reader = PdfReader(path)
    require(len(reader.pages) == 1, "PDF must have one page")
    page = reader.pages[0]
    expected_box = (0, 0, float(SIZE_PT), float(SIZE_PT))
    for box in (page.mediabox, page.cropbox):
        np.testing.assert_allclose(list(map(float, box)), expected_box, atol=POINT_TOL, rtol=0)
    require(page.rotation == 0 and float(page.get("/UserUnit", 1)) == 1, "PDF is scaled or rotated")
    operators = Counter(op for _, op in page.get_contents().operations)
    require(operators[b"S"] == 92 and operators[b"B"] == 61 and operators[b"c"] > 0,
            "PDF path paint counts do not match the map")
    require(not operators[b"Do"] and not operators[b"INLINE IMAGE"], "PDF includes raster content")
    print("PDF: one 2500 × 2500 mm vector page, 92 strokes and 61 filled/stroked paths")


def compare_preview(path, reference_path):
    with Image.open(path) as png, Image.open(reference_path) as reference:
        require(png.size == (2000, 2000), "expected a 2000 × 2000 px preview")
        rgb = np.asarray(png.convert("RGB"), dtype=np.int16)
        ref = np.asarray(reference.convert("RGB").resize(png.size, Image.Resampling.LANCZOS),
                         dtype=np.int16)
    require(np.all(rgb[1000, 1000] == 255), "preview background is not pure white")
    delta = np.abs(rgb - ref)
    identical = (delta.max(axis=2) == 0).mean()
    close = (delta.max(axis=2) <= 15).mean()
    require(close >= 0.98 and delta.mean() <= 2, "preview differs substantially from the reference")
    print(f"PNG vs reference: {identical:.2%} exact RGB pixels, "
          f"{close:.2%} within 15 levels/channel, mean difference {delta.mean():.3f}/255")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PNG)
    args = parser.parse_args()
    features = map_features(load_map(args.input))
    verify_eps(args.output_dir / EPS_NAME, features)
    verify_pdf(args.output_dir / PDF_NAME)
    compare_preview(args.output_dir / PNG_NAME, args.reference)


if __name__ == "__main__":
    main()
