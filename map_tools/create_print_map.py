"""Render the existing 2.5 m campus map as a full-size vector EPS and previews."""

import argparse
from dataclasses import dataclass
from decimal import Decimal
import json
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.path import Path as VectorPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOURCE_DIR = DATA_DIR / "maps"
DEFAULT_INPUT = SOURCE_DIR / "KAIST_model_features_2p5m.json"
DEFAULT_OUTPUT = DATA_DIR / "print" / "campus"
EPS_NAME = "KAIST_map_PRINT_2500x2500mm.eps"
PNG_NAME = "KAIST_map_PRINT_preview.png"
PDF_NAME = "KAIST_map_PRINT_preview.pdf"

SIZE_M = 2.5
SIZE_MM = Decimal("2500")
PT_PER_MM = Decimal(72) / Decimal("25.4")
SIZE_PT = SIZE_MM * PT_PER_MM
PT_PER_M = float(1000 * PT_PER_MM)
SIZE_IN = float(SIZE_MM / Decimal("25.4"))
REFERENCE_IN = 10
ROAD_OUTLINES = {
    "service": "#ffffff",
    "unclassified": "#fdfdfd",
    "secondary": "#fdfdfd",
}
BUILDING_EDGE = "#283240"
GATE_COLOR = "#dd343f"
GATE_RADIUS_M = 0.016


@dataclass(frozen=True)
class Feature:
    kind: str
    id: str
    path: VectorPath
    fill: str | None = None
    stroke: str | None = None
    width_ref_pt: float = 0
    rounded: bool = False


def load_map(path: Path) -> dict:
    """Read the model JSON and reject inputs with a different print extent."""
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    if (meta["model_width_m"], meta["model_height_m"]) != (SIZE_M, SIZE_M):
        raise ValueError("input must describe a 2.5 m by 2.5 m model")
    if (len(data["buildings"]), len(data["roads"])) != (60, 46):
        raise ValueError("expected the KAIST model with 60 buildings and 46 roads")
    return data


def map_features(data: dict) -> list[Feature]:
    """Build the draw order from the original reference renderer and JSON geometry."""
    roads = sorted(data["roads"], key=lambda road: tuple(ROAD_OUTLINES).index(road["highway"]))
    road_paths = [(road, VectorPath(road["centerline_xy_m"])) for road in roads]
    features = [
        Feature("road-outline", road["id"], path, stroke=ROAD_OUTLINES[road["highway"]],
                width_ref_pt=road["display_linewidth_pt"] + 2.2, rounded=True)
        for road, path in road_paths
    ]
    features.extend(
        Feature("road", road["id"], path, stroke=road["display_color"],
                width_ref_pt=road["display_linewidth_pt"], rounded=True)
        for road, path in road_paths
    )
    features.extend(
        Feature("building", building["id"], VectorPath(building["polygon_xy_m"], closed=True),
                fill=building["color"], stroke=BUILDING_EDGE, width_ref_pt=0.55)
        for building in data["buildings"]
    )
    gate_x, gate_y = data["landmarks"]["main_gate"]["xy_m"]
    gate = VectorPath.unit_circle().transformed(
        Affine2D().scale(GATE_RADIUS_M).translate(gate_x, gate_y))
    features.append(Feature("gate", "main_gate", gate, fill=GATE_COLOR,
                            stroke="#ffffff", width_ref_pt=0.7))
    return features


def ps_number(value) -> str:
    """Write decimal PostScript numbers with sub-micrometer print precision."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def ps_color(hex_color: str) -> str:
    return " ".join(ps_number(channel) for channel in to_rgb(hex_color)) + " setrgbcolor"


def write_eps(destination: Path, features: list[Feature]) -> None:
    """Write point-based, single-page EPS with no device-specific page commands."""
    edge = ps_number(SIZE_PT)
    bounds = ceil(SIZE_PT)
    lines = [
        "%!PS-Adobe-3.0 EPSF-3.0",
        f"%%Title: {EPS_NAME}",
        f"%%BoundingBox: 0 0 {bounds} {bounds}",
        f"%%HiResBoundingBox: 0 0 {edge} {edge}",
        "%%LanguageLevel: 2",
        "%%Pages: 1",
        "%%EndComments",
        "gsave",
        f"newpath 0 0 moveto {edge} 0 lineto {edge} {edge} lineto 0 {edge} lineto closepath clip newpath",
        f"1 1 1 setrgbcolor 0 0 {edge} {edge} rectfill",
    ]
    operators = {
        VectorPath.MOVETO: "moveto", VectorPath.LINETO: "lineto",
        VectorPath.CURVE4: "curveto", VectorPath.CLOSEPOLY: "closepath",
    }
    for feature in features:
        lines.extend((f"%%MapFeature: {feature.kind} {feature.id}", "newpath"))
        for vertices, code in feature.path.iter_segments(simplify=False, curves=True):
            if code == VectorPath.CLOSEPOLY:
                lines.append("closepath")
            else:
                coordinates = " ".join(ps_number(v * PT_PER_M) for v in vertices)
                lines.append(f"{coordinates} {operators[code]}")
        if feature.fill:
            lines.extend(("gsave", ps_color(feature.fill), "fill", "grestore"))
        if feature.stroke:
            width_pt = feature.width_ref_pt * SIZE_IN / REFERENCE_IN
            cap_join = 1 if feature.rounded else 0
            lines.extend((ps_color(feature.stroke), f"{ps_number(width_pt)} setlinewidth",
                          f"{cap_join} setlinecap", f"{cap_join} setlinejoin", "stroke"))
    lines.extend(("%%MapFeaturesEnd", "grestore", "showpage", "%%Trailer", "%%EOF", ""))
    destination.write_text("\n".join(lines), encoding="ascii", newline="\n")


def preview_figure(features: list[Feature], size_in: float) -> Figure:
    """Draw the same vector paths, with relative stroke widths at either size."""
    figure = Figure(figsize=(size_in, size_in), dpi=200, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_axes([0, 0, 1, 1], xlim=(0, SIZE_M), ylim=(0, SIZE_M), aspect="equal")
    axes.set_axis_off()
    for feature in features:
        axes.add_patch(PathPatch(
            feature.path, facecolor=feature.fill or "none", edgecolor=feature.stroke or "none",
            linewidth=feature.width_ref_pt * size_in / REFERENCE_IN,
            capstyle="round" if feature.rounded else "butt",
            joinstyle="round" if feature.rounded else "miter",
        ))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    features = map_features(load_map(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_eps(args.output_dir / EPS_NAME, features)
    preview_figure(features, REFERENCE_IN).savefig(args.output_dir / PNG_NAME, dpi=200)
    preview_figure(features, SIZE_IN).savefig(args.output_dir / PDF_NAME)
    for name in (EPS_NAME, PNG_NAME, PDF_NAME):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
