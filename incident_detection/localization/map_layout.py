"""Plan AprilTags and photo slots on the printed color map, and make everything needed to paste them.

The ordered color map (data/maps/KAIST_model_reference_2p5m.png, printed 2.5 m x 2.5 m) stays as it is. The grayscale
camera cannot tell its building colors apart, so AprilTags are pasted on top of it, only where two
printed edges meet at a right angle (90 deg within TagConfig.max_corner_error_deg) and are both
longer than the tag paper, so pushing the paper into that corner fixes its position and rotation:
  - building tags: in the outer corners of buildings, inside the dark outline; spread over the map
    (farthest-point choice, at least TagConfig.min_spacing_m apart),
  - road tags (ids after the building tags): in the corners of the open map between roads, where
    two road edges (or a road edge and a building outline) meet; added one by one where they cover
    the most road not yet near a tag, for flights that follow roads,
  - never on or next to a photo slot.
Photo slots are rectangles centered over large buildings (the photo shows something happening at
that building), lined up with its walls. They may hang over open map, never over another building.

Outputs in data/maps/:
    apriltag_map.json          pose of every tag on the map (localizer.py reads it)
    slots.json                 photo slots and the example photo in each
    layout_guide.png           the pasting plan: map with tag ids and slots
    map_with_tags.png          the finished map in color (0.5 mm per pixel)
    map_with_tags_gray.png     the same, converted to grayscale
    map_with_photos.png        finished map plus the example photos in the slots, color
    map_with_photos_gray.png   the same, grayscale (roughly what the grayscale camera sees)
and in data/print/markers/:
    tags/tag_NN.pdf            one tag per A4 page at exact size, with a placement picture
    all_tags.pdf               all tag pages in one file
    photos/slot_N.pdf          the example photos at slot size

Usage:
    python map_layout.py
"""
import hashlib
import json
import logging
import math
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patheffects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrow, Polygon as PolygonPatch, Rectangle
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

import _paths  # noqa: F401  (LLM_detects on sys.path, for the test photos)
from config import Config  # LLM_detects: where the test photos are
from dataset import find_photo
from photo_finder import ARUCO_DICTS
from settings import MapConfig, SlotConfig, TagConfig
from image_io import read_image, write_image

PX_PER_MM = 2.0  # render scale of the map images (0.5 mm per pixel)
A4_MM = (210.0, 297.0)
COMPASS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
KOREAN_FONTS = ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR")
GATE_MARKER_RADIUS_M = 0.016  # red main-gate circle drawn on the reference map
MAP_BORDER_M = 0.005  # keep tags this far inside the paper edge
FIGURE_PT = 720  # the reference map is a 10 in (720 pt) figure for the whole map; road widths are in pt
SLOT_BORDER_M = 0.2  # keep slots off the outer 20 cm of the map


# ---------- geometry ----------

def right_of(up: np.ndarray) -> np.ndarray:
    """+x direction of something whose top points `up`, seen from above with north up."""
    return np.array([up[1], -up[0]])


def rect_corners(center, up, width: float, height: float) -> np.ndarray:
    """Corners TL, TR, BR, BL of a width x height rectangle whose top side faces `up`."""
    c = np.asarray(center, float)
    r = right_of(np.asarray(up, float)) * width / 2
    u = np.asarray(up, float) * height / 2
    return np.array([c - r + u, c + r + u, c + r - u, c - r - u])


def compass(v) -> str:
    return COMPASS[round(math.degrees(math.atan2(v[1], v[0])) / 45) % 8]


def to_px(xy, map_cfg: MapConfig, px_per_mm: float = PX_PER_MM) -> np.ndarray:
    """Map meters -> image pixel coordinates (OpenCV convention: pixel centers sit on integers)."""
    xy = np.asarray(xy, float)
    s = px_per_mm * 1000
    return np.stack([xy[..., 0] * s - 0.5, (map_cfg.size_m - xy[..., 1]) * s - 0.5], axis=-1)


@dataclass
class TagPlacement:
    id: int
    center: np.ndarray  # map meters
    up: np.ndarray  # unit vector: the top of the printed tag points this way on the map
    corner_xy: np.ndarray  # the printed corner the paper is pushed into
    edge_m: float  # shorter of the two printed edges that form that corner
    corner_error_deg: float
    kind: str = "building"  # "building" (outer building corner) or "road" (corner of the open map between roads)
    building: int = -1  # index into features["buildings"] for building tags
    place: str = ""  # what forms a road tag's corner, for its print page

    def corners(self, side_m: float) -> np.ndarray:
        return rect_corners(self.center, self.up, side_m, side_m)

    @property
    def yaw_deg(self) -> float:
        r = right_of(self.up)
        return math.degrees(math.atan2(r[1], r[0]))

    @property
    def corner_name(self) -> str:
        return compass(self.corner_xy - self.center)


def square_in_corner(p, prev, nxt, paper: float, inset: float, max_error_deg: float):
    """The tag paper pushed into the corner at vertex p, whose edges run toward prev and nxt.

    Returns (paper corner, u, v, shorter edge length, angle error), with u along the longer edge
    (the one to line the paper up with) and v square to it, or None when the corner is not square
    enough or an edge is shorter than the paper. inset moves the paper off both edges."""
    e1, e2 = prev - p, nxt - p
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if min(l1, l2) < paper:
        return None
    angle = math.degrees(math.acos(np.clip(e1 @ e2 / (l1 * l2), -1, 1)))
    if abs(angle - 90) > max_error_deg:
        return None
    (eu, lu), (ew, _) = ((e1, l1), (e2, l2)) if l1 >= l2 else ((e2, l2), (e1, l1))
    u = eu / lu
    v = ew - (ew @ u) * u
    v /= np.linalg.norm(v)
    return p + inset * (u + v), u, v, min(l1, l2), abs(angle - 90)


def square_against_edges(p, prev, nxt, paper: float, min_angle_deg: float, max_angle_deg: float):
    """The tag paper laid along the longer of the two edges at vertex p and slid toward the corner
    until it touches the other edge: at the corner itself when the angle is 90 deg or more, and short
    of it (its far corner touching the other edge) when the angle is sharper. One unique place either way.

    Returns (paper corner, u, v, shorter edge length, angle) with u along the longer edge and v
    square to it, or None when the angle is out of range or an edge is too short to rest against."""
    e1, e2 = prev - p, nxt - p
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if min(l1, l2) < 1e-6:
        return None
    angle = math.degrees(math.acos(np.clip(e1 @ e2 / (l1 * l2), -1, 1)))
    if not min_angle_deg <= angle <= max_angle_deg:
        return None
    (eu, lu), (ew, lw) = ((e1, l1), (e2, l2)) if l1 >= l2 else ((e2, l2), (e1, l1))
    u = eu / lu
    v = ew - (ew @ u) * u
    v /= np.linalg.norm(v)
    shift = paper / math.tan(math.radians(angle)) if angle < 90 else 0.0
    stop = paper / math.sin(math.radians(angle)) if angle < 90 else 0.02  # length of the other edge needed
    if lu < shift + paper or lw < stop:
        return None
    return p + shift * u, u, v, min(lu, lw), angle


def upright(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Of the four ways to turn a square tag in a corner, the one whose top points most to the north."""
    return max((u, v, -u, -v), key=lambda d: d[1])


@dataclass
class Slot:
    number: int
    center: np.ndarray
    up: np.ndarray  # top of the slot (and of the photo in it); faces north-ish
    building: int
    cover: float  # fraction of the slot that lies on its building
    photo_id: str = ""  # example photo, a photo_unlabeled id

    def corners(self, cfg: SlotConfig) -> np.ndarray:
        return rect_corners(self.center, self.up, cfg.width_mm / 1000, cfg.height_mm / 1000)


@dataclass
class Piece:
    """A printed photo, centered in its slot: corners (TL, TR, BR, BL, map meters) of each layer."""
    outer: np.ndarray  # cut line = outer edge of the white margin
    frame: np.ndarray  # outer edge of the black frame (what the camera finds)
    picture: np.ndarray  # the photo itself
    size_mm: tuple[float, float]  # outer width, height
    picture_mm: tuple[float, float]


def photo_piece(slot: Slot, photo_hw: tuple[int, int], cfg: SlotConfig, frame_mm: float | None = None,
                white_mm: float | None = None, bar_mm: float | None = None) -> Piece:
    """The picture keeps its aspect ratio and is as large as the slot allows with frame, bottom bar
    and margin; the whole piece is centered in the slot."""
    f = (cfg.frame_mm if frame_mm is None else frame_mm) / 1000
    m = (cfg.white_margin_mm if white_mm is None else white_mm) / 1000
    b = (cfg.bottom_bar_mm if bar_mm is None else bar_mm) / 1000
    h, w = photo_hw
    s = min((cfg.width_mm / 1000 - 2 * (f + m)) / w, (cfg.height_mm / 1000 - (f + b + 2 * m)) / h)
    iw, ih = w * s, h * s
    c, up = np.asarray(slot.center, float), np.asarray(slot.up, float)
    picture_center = c + up * (b - f) / 2  # the bar pushes the picture up
    return Piece(rect_corners(c, up, iw + 2 * (f + m), ih + f + b + 2 * m),
                 rect_corners(c, up, iw + 2 * f, ih + f + b),
                 rect_corners(picture_center, up, iw, ih),
                 ((iw + 2 * (f + m)) * 1000, (ih + f + b + 2 * m) * 1000), (iw * 1000, ih * 1000))


def choose_spread(items: list, xy: list[np.ndarray], quality: list[float], first: int,
                  spacing: float, max_n: int) -> list:
    """Farthest-point choice: repeatedly take the item farthest from everything taken so far.

    Among items nearly as far (85% of the best distance), the higher quality one wins.
    Stops when the best distance drops below `spacing` or max_n items are taken."""
    xy = np.asarray(xy)
    taken = [first]
    dist = np.linalg.norm(xy - xy[first], axis=1)
    while len(taken) < max_n:
        best = dist.max()
        if best < spacing:
            break
        near_best = np.flatnonzero(dist >= 0.85 * best)
        pick = max(near_best, key=lambda i: quality[i])
        taken.append(int(pick))
        dist = np.minimum(dist, np.linalg.norm(xy - xy[pick], axis=1))
    return [items[i] for i in taken]


# ---------- layout ----------

def load_buildings(map_cfg: MapConfig) -> tuple[dict, list[Polygon]]:
    features = json.loads(map_cfg.features.read_text(encoding="utf-8"))
    polys = [orient(Polygon(b["polygon_xy_m"]), 1.0) for b in features["buildings"]]  # counter-clockwise
    return features, polys


def wall_directions(poly: Polygon, min_len_m: float = 0.03) -> list[float]:
    """Distinct directions (radians, in (-90, 90] deg) of a building's walls, longest walls first."""
    pts = np.asarray(poly.exterior.coords)
    walls = sorted(((np.linalg.norm(b - a), b - a) for a, b in zip(pts[:-1], pts[1:])), key=lambda e: -e[0])
    dirs = []
    for length, d in walls:
        if length < min_len_m:
            break
        for theta in (math.atan2(d[1], d[0]), math.atan2(d[1], d[0]) + math.pi / 2):  # along and across
            theta = (theta + math.pi / 2) % math.pi - math.pi / 2  # [-90, 90)
            if theta == -math.pi / 2:
                theta = math.pi / 2
            if all(abs(math.sin(theta - t)) > math.sin(math.radians(3)) for t in dirs):
                dirs.append(theta)
    return dirs


def find_slots(polys: list[Polygon], map_cfg: MapConfig, cfg: SlotConfig, gate_xy,
               step_m: float = 0.01) -> list[Slot]:
    """Best slot per building: centered over it, wall-aligned, covering it as much as possible.

    Most buildings are only 7-12 cm wide on the 2.5 m map, narrower than a photo the camera can
    read, so a slot may hang over open map next to its building. It never touches another
    building or the main-gate marker, and keeps margin_mm from them."""
    w, h = cfg.width_mm / 1000, cfg.height_mm / 1000
    margin = cfg.margin_mm / 1000
    lo, hi = SLOT_BORDER_M, map_cfg.size_m - SLOT_BORDER_M
    gate = Point(gate_xy).buffer(GATE_MARKER_RADIUS_M + margin)
    found = []
    for bi, poly in enumerate(polys):
        if poly.area < cfg.min_cover * w * h:
            continue
        inside = prep(poly)
        others = [p.buffer(margin) for j, p in enumerate(polys) if j != bi and p.distance(poly) < w + margin]
        coords = np.asarray(poly.exterior.coords)
        best = None
        for theta in wall_directions(poly):
            up = np.array([-math.sin(theta), math.cos(theta)])  # y >= 0: top faces north-ish
            right = right_of(up)
            a, b = coords @ right, coords @ up
            for ca in np.arange(a.min(), a.max() + 1e-9, step_m):
                for cb in np.arange(b.min(), b.max() + 1e-9, step_m):
                    c = ca * right + cb * up
                    if not (lo <= c[0] <= hi and lo <= c[1] <= hi) or not inside.contains(Point(c)):
                        continue
                    rect = Polygon(rect_corners(c, up, w, h))
                    cover = rect.intersection(poly).area / rect.area
                    if cover < cfg.min_cover or (best and cover <= best[0]):
                        continue
                    if rect.intersects(gate) or any(o.intersects(rect) for o in others):
                        continue
                    best = (cover, c, up)
        if best:
            found.append(Slot(0, best[1], best[2], bi, best[0]))
    return found


def choose_slots(polys, map_cfg: MapConfig, cfg: SlotConfig, gate_xy) -> list[Slot]:
    # Best-covered buildings first, skipping any closer than min_spacing_m to one already taken
    chosen = []
    for s in sorted(find_slots(polys, map_cfg, cfg, gate_xy), key=lambda s: -s.cover):
        if all(np.linalg.norm(s.center - c.center) >= cfg.min_spacing_m for c in chosen):
            chosen.append(s)
        if len(chosen) == cfg.count:
            break
    else:
        raise SystemExit(f"Only {len(chosen)} slots of {cfg.width_mm} x {cfg.height_mm} mm fit at least "
                         f"{cfg.min_spacing_m} m apart; lower SlotConfig.count, size or min_spacing_m")
    chosen.sort(key=lambda s: (-s.center[1], s.center[0]))  # number north to south
    for n, slot in enumerate(chosen, 1):
        slot.number = n
        slot.photo_id = cfg.example_ids[(n - 1) % len(cfg.example_ids)]
    return chosen


def find_tag_spots(polys, map_cfg: MapConfig, cfg: TagConfig, keepouts: list[Polygon],
                   gate_xy) -> list[TagPlacement]:
    """Every square building corner where the whole tag paper fits inside the building's outline."""
    inset = map_cfg.outline_width_mm / 2000
    paper = cfg.paper_mm / 1000
    gate = Point(gate_xy).buffer(GATE_MARKER_RADIUS_M + 0.005)
    lo, hi = MAP_BORDER_M, map_cfg.size_m - MAP_BORDER_M
    spots = []
    for bi, poly in enumerate(polys):
        # The paper may cover at most the inner half of the dark outline (the polygon edge itself)
        fits = prep(poly.buffer(1e-6, join_style="mitre"))
        others = [p for j, p in enumerate(polys) if j != bi and p.distance(poly) < paper]
        pts = np.asarray(poly.exterior.coords)[:-1]
        for i in range(len(pts)):
            found = square_in_corner(pts[i], pts[i - 1], pts[(i + 1) % len(pts)], paper, inset,
                                     cfg.max_corner_error_deg)  # inset: paper just inside the outline
            if found is None:
                continue
            origin, u, v, edge, error = found
            square = Polygon([origin, origin + paper * u, origin + paper * (u + v), origin + paper * v])
            sx, sy = np.asarray(square.exterior.coords).T
            if (not fits.contains(square) or sx.min() < lo or sy.min() < lo or sx.max() > hi or sy.max() > hi
                    or square.intersects(gate) or any(o.intersects(square) for o in others)
                    or any(k.intersects(square) for k in keepouts)):
                continue
            spots.append(TagPlacement(-1, origin + paper / 2 * (u + v), upright(u, v), pts[i].copy(), edge, error,
                                      kind="building", building=bi))
    return spots


def choose_tags(polys, map_cfg: MapConfig, cfg: TagConfig, slots: list[Slot], slot_cfg: SlotConfig,
                gate_xy) -> tuple[list[TagPlacement], int]:
    keepouts = [Polygon(s.corners(slot_cfg)).buffer(slot_cfg.margin_mm / 1000, join_style="mitre")
                for s in slots]
    spots = find_tag_spots(polys, map_cfg, cfg, keepouts, gate_xy)
    if not spots:
        raise SystemExit("No building corner fits a tag; make TagConfig.size_mm smaller")
    mid = np.array([map_cfg.size_m / 2] * 2)
    quality = [s.edge_m - s.corner_error_deg * 0.01 for s in spots]
    first = int(np.argmin([np.linalg.norm(s.center - mid) for s in spots]))
    tags = choose_spread(spots, [s.center for s in spots], quality, first, cfg.min_spacing_m, cfg.max_tags)
    # Ids run west to east inside 25 cm bands from north to south, so neighbors have nearby ids
    tags.sort(key=lambda t: (math.floor((map_cfg.size_m - t.center[1]) / 0.25), t.center[0]))
    for k, t in enumerate(tags):
        t.id = cfg.first_id + k
    return tags, len(spots)


def road_shapes(features: dict, map_cfg: MapConfig) -> list[Polygon]:
    """The gray road strokes as printed: centerlines widened by their drawn line width (in pt on the
    reference figure, which spans the whole map in FIGURE_PT points). Matches the print within 0.1 mm."""
    m_per_pt = map_cfg.size_m / FIGURE_PT
    return [LineString(r["centerline_xy_m"]).buffer(r["display_linewidth_pt"] * m_per_pt / 2, quad_segs=16)
            for r in features["roads"]]


def corner_place(p, features: dict, roads: list[Polygon], polys: list[Polygon]) -> str:
    """Names of the roads and buildings whose printed edges form the corner at p."""
    here = Point(p)
    near = [r["name"] for r, shape in zip(features["roads"], roads) if shape.distance(here) < 0.001]
    names = list(dict.fromkeys(n for n in near if n))
    unnamed = sum(not n for n in near)
    if unnamed:
        names.append("an unnamed road" if unnamed == 1 else f"{unnamed} unnamed roads")
    names += [building_label(b) for b, poly in zip(features["buildings"], polys) if poly.distance(here) < 0.002]
    return "corner between " + " and ".join(names) if names else "corner on the open map"


def find_road_spots(features: dict, polys, map_cfg: MapConfig, cfg: TagConfig, keepouts: list[Polygon],
                    gate_xy) -> list[TagPlacement]:
    """Corners of the open map between roads, where two printed road edges (or a road edge and a
    building outline) meet at road_corner_angle_deg: the paper lies along the longer edge, pushed
    into the corner until it touches the other edge (square_against_edges)."""
    inset = map_cfg.outline_width_mm / 2000
    paper = cfg.paper_mm / 1000
    roads = road_shapes(features, map_cfg)
    buildings = unary_union([p.buffer(inset, join_style="mitre") for p in polys])  # outline stays visible
    free = box(0, 0, map_cfg.size_m, map_cfg.size_m).difference(unary_union(roads)).difference(buildings)
    fits = prep(free.buffer(0.0006))  # corners come from a 0.5 mm simplified outline
    gate = Point(gate_xy).buffer(GATE_MARKER_RADIUS_M + 0.005)
    lo, hi = MAP_BORDER_M, map_cfg.size_m - MAP_BORDER_M
    spots = []
    for part in getattr(free, "geoms", [free]):
        if part.geom_type != "Polygon":
            continue
        simple = part.simplify(0.0005)  # each straight printed edge becomes one segment
        for ring in [simple.exterior, *simple.interiors]:
            pts = np.asarray(ring.coords)[:-1]
            for i in range(len(pts)):
                found = square_against_edges(pts[i], pts[i - 1], pts[(i + 1) % len(pts)], paper,
                                             *cfg.road_corner_angle_deg)
                if found is None:
                    continue
                origin, u, v, edge, angle = found
                error = abs(angle - 90)
                square = Polygon([origin, origin + paper * u, origin + paper * (u + v), origin + paper * v])
                sx, sy = np.asarray(square.exterior.coords).T
                if (not fits.contains(square) or sx.min() < lo or sy.min() < lo or sx.max() > hi or sy.max() > hi
                        or square.intersects(gate) or any(k.intersects(square) for k in keepouts)):
                    continue
                spots.append(TagPlacement(-1, origin + paper / 2 * (u + v), upright(u, v), pts[i].copy(), edge, error,
                                          kind="road", place=corner_place(pts[i], features, roads, polys)))
    return spots


def road_points(features: dict, step_m: float) -> np.ndarray:
    """Points every step_m along all road centerlines."""
    pts = []
    for r in features["roads"]:
        line = LineString(r["centerline_xy_m"])
        pts += [line.interpolate(d).coords[0] for d in np.arange(0, line.length, step_m)]
    return np.array(pts)


def choose_road_tags(features, polys, map_cfg: MapConfig, cfg: TagConfig, slots: list[Slot], slot_cfg: SlotConfig,
                     gate_xy, existing: list[TagPlacement]) -> tuple[list[TagPlacement], dict]:
    """Greedy: add the road corner that brings the most road within road_cover_radius_m of a tag,
    until the next one would add less than min_road_gain_m. Ids continue after the existing tags."""
    paper = cfg.paper_mm / 1000
    keepouts = [Polygon(s.corners(slot_cfg)).buffer(slot_cfg.margin_mm / 1000, join_style="mitre") for s in slots]
    keepouts += [Polygon(t.corners(paper)).buffer(0.01) for t in existing]
    spots = find_road_spots(features, polys, map_cfg, cfg, keepouts, gate_xy)
    step = 0.02
    pts = road_points(features, step)
    near = lambda c: np.linalg.norm(pts - c, axis=1) <= cfg.road_cover_radius_m
    centers = [t.center for t in existing]
    covered = np.zeros(len(pts), bool)
    for c in centers:
        covered |= near(c)
    before = float(covered.mean())
    reach = [near(s.center) for s in spots]
    chosen = []
    while len(chosen) < cfg.max_road_tags:
        best, best_gain = None, 0.0
        for k, s in enumerate(spots):
            if any(np.linalg.norm(s.center - c) < cfg.min_spacing_m for c in centers):
                continue
            gain = float((reach[k] & ~covered).sum()) * step
            if gain > best_gain:
                best, best_gain = k, gain
        if best is None or best_gain < cfg.min_road_gain_m:
            break
        chosen.append(spots[best])
        centers.append(spots[best].center)
        covered |= reach[best]
    chosen.sort(key=lambda t: (math.floor((map_cfg.size_m - t.center[1]) / 0.25), t.center[0]))
    first = max((t.id for t in existing), default=cfg.first_id - 1) + 1
    for k, t in enumerate(chosen):
        t.id = first + k
    return chosen, {"road_spots": len(spots), "road_covered_before": before, "road_covered_after": float(covered.mean())}


# ---------- rendering ----------

def tag_image(tag_id: int, cfg: TagConfig, cell_px: int) -> np.ndarray:
    """Grayscale tag with its white quiet zone: (8 + 2 * quiet) cells of cell_px pixels each side."""
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[cfg.family]),
                                           tag_id, 8 * cell_px, borderBits=1)
    q = cfg.quiet_zone_cells * cell_px
    return cv2.copyMakeBorder(marker, q, q, q, q, cv2.BORDER_CONSTANT, value=255)


def framed_photo(path, size_px: tuple[int, int], frame_px: int, bar_px: int, white_px: int, gray: bool) -> np.ndarray:
    """The photo resized to size_px, then a black frame (bar_px thick at the bottom), then a white margin."""
    img = read_image(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    img = cv2.resize(img, size_px, interpolation=cv2.INTER_AREA)
    black, white = (0, 255) if gray else ((0, 0, 0), (255, 255, 255))
    img = cv2.copyMakeBorder(img, frame_px, bar_px, frame_px, frame_px, cv2.BORDER_CONSTANT, value=black)
    return cv2.copyMakeBorder(img, white_px, white_px, white_px, white_px, cv2.BORDER_CONSTANT, value=white)


def paste(canvas: np.ndarray, src: np.ndarray, quad_px: np.ndarray) -> None:
    """Warp `src` so its outer corners TL, TR, BR, BL land on quad_px, blending the edges, in place."""
    if canvas.ndim == 3 and src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    elif canvas.ndim == 2 and src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    h, w = src.shape[:2]
    src_quad = np.float32([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    x0, y0 = np.maximum(np.floor(quad_px.min(0)).astype(int) - 1, 0)
    x1 = min(int(np.ceil(quad_px[:, 0].max())) + 2, canvas.shape[1])
    y1 = min(int(np.ceil(quad_px[:, 1].max())) + 2, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m = cv2.getPerspectiveTransform(src_quad, np.float32(quad_px - [x0, y0]))
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpPerspective(src, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpPerspective(np.ones((h, w), np.float32), m, size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    roi = canvas[y0:y1, x0:x1]
    if roi.ndim == 3:
        alpha = alpha[..., None]
    roi[:] = np.round(roi * (1 - alpha) + warped * alpha).astype(canvas.dtype)


def render_base(map_cfg: MapConfig, gray: bool = False) -> np.ndarray:
    img = read_image(map_cfg.image, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    n = round(map_cfg.size_m * 1000 * PX_PER_MM)
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_LINEAR)


def render_tags(canvas, tags: list[TagPlacement], map_cfg: MapConfig, cfg: TagConfig) -> None:
    cell_px = max(1, round(cfg.size_mm / 8 * PX_PER_MM))
    for t in tags:
        paste(canvas, tag_image(t.id, cfg, cell_px), to_px(t.corners(cfg.paper_mm / 1000), map_cfg))


def render_photos(canvas, slots: list[Slot], map_cfg: MapConfig, cfg: SlotConfig,
                  frame_mm: float | None = None, white_mm: float | None = None) -> None:
    frame_mm = cfg.frame_mm if frame_mm is None else frame_mm
    white_mm = cfg.white_margin_mm if white_mm is None else white_mm
    for s in slots:
        path = photo_path(s.photo_id)
        piece = photo_piece(s, read_image(path, cv2.IMREAD_GRAYSCALE).shape, cfg, frame_mm, white_mm)
        size_px = tuple(round(v * PX_PER_MM) for v in piece.picture_mm)
        img = framed_photo(path, size_px, round(frame_mm * PX_PER_MM), round(cfg.bottom_bar_mm * PX_PER_MM),
                           round(white_mm * PX_PER_MM), gray=canvas.ndim == 2)
        paste(canvas, img, to_px(piece.outer, map_cfg))


def photo_path(photo_id: str):
    return find_photo(Config().image_dir, photo_id)


def render_finished_map(map_cfg: MapConfig, tag_cfg: TagConfig, slot_cfg: SlotConfig, tags, slots,
                        gray: bool, with_photos: bool, frame_mm: float | None = None,
                        white_mm: float | None = None) -> np.ndarray:
    """The printed map with tags (and photos) pasted on, at PX_PER_MM. Gray converts the color render."""
    canvas = render_base(map_cfg)
    render_tags(canvas, tags, map_cfg, tag_cfg)
    if with_photos:
        render_photos(canvas, slots, map_cfg, slot_cfg, frame_mm, white_mm)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if gray else canvas


# ---------- files ----------

def building_label(b: dict) -> str:
    if b["name"]:
        return b["name"]
    return f"unnamed building ({b['id']})"


def place_label(t: TagPlacement, features: dict) -> str:
    """Where a tag goes, in words: "SW corner of <building>" or "corner between <road> and <road>"."""
    if t.kind == "road":
        return t.place
    b = features["buildings"][t.building]
    ref = f" {b['ref']}" if b["ref"] and b["ref"] not in b["name"] else ""
    return f"{t.corner_name} corner of {building_label(b)}{ref}"


def write_json(map_cfg, tag_cfg, slot_cfg, features, tags, slots) -> None:
    size, paper = tag_cfg.size_mm / 1000, tag_cfg.paper_mm / 1000
    r6 = lambda a: np.round(np.asarray(a, float), 6).tolist()
    tag_doc = {
        "family": tag_cfg.family,
        "opencv_dictionary": f"DICT_APRILTAG_{tag_cfg.family.removeprefix('tag')}",
        "tag_size_m": size,
        "paper_size_m": paper,
        "map_frame": "meters on the printed map; origin at its lower-left corner; +x east (right), +y north (up); "
                     "z = 0 on the map surface",
        "corner_order": "corners_xy_m are the black square's corners in OpenCV ArucoDetector order: top-left, "
                        "top-right, bottom-right, bottom-left of the printed tag. pupil-apriltags returns the same "
                        "corners as top-right, top-left, bottom-left, bottom-right (checked with pupil-apriltags "
                        "1.0.4); both libraries give the same ids.",
        "yaw_deg": "angle from map +x to the tag's +x (left to right across the printed tag), counter-clockwise",
        "coordinate_frame": "kaist_map",
        "map_size_m": [map_cfg.size_m, map_cfg.size_m],
        "reference_sha256": hashlib.sha256(map_cfg.features.read_bytes()).hexdigest(),
        "print_scale": "physical print width divided by model width; scale the map and all tags together",
        "tags": [{
            "id": t.id,
            "center_xy_m": r6(t.center),
            "yaw_deg": round(t.yaw_deg, 3),
            "corners_xy_m": r6(t.corners(size)),
            "paper_corners_xy_m": r6(t.corners(paper)),
            "kind": t.kind,
            "place": place_label(t, features),
            "corner_xy_m": r6(t.corner_xy),
            "building_index": t.building if t.kind == "building" else None,
            "building_osm_id": features["buildings"][t.building]["id"] if t.kind == "building" else None,
        } for t in tags],
    }
    (map_cfg.out_dir / "apriltag_map.json").write_text(json.dumps(tag_doc, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")

    slot_doc = {
        "slot_size_m": [slot_cfg.width_mm / 1000, slot_cfg.height_mm / 1000],
        "frame_mm": slot_cfg.frame_mm,
        "bottom_bar_mm": slot_cfg.bottom_bar_mm,
        "white_margin_mm": slot_cfg.white_margin_mm,
        "map_frame": tag_doc["map_frame"],
        "note": "corners are TL, TR, BR, BL with the top facing `up_xy`. Each example photo is printed as large "
                "as fits (picture, black frame that is bottom_bar_mm thick under the picture, white margin), "
                "and the whole piece is centered in the slot, its top toward `up_xy`; the bar shifts the "
                "picture up a little. example_frame_corners_xy_m is the black frame's outer edge, the outline "
                "the camera finds; example_picture_corners_xy_m is what the LLM gets.",
        "slots": [],
    }
    for s in slots:
        piece = photo_piece(s, read_image(photo_path(s.photo_id), cv2.IMREAD_GRAYSCALE).shape, slot_cfg)
        slot_doc["slots"].append({
            "slot": s.number,
            "center_xy_m": r6(s.center),
            "up_xy": r6(s.up),
            "corners_xy_m": r6(s.corners(slot_cfg)),
            "building_index": s.building,
            "building_name": features["buildings"][s.building]["name"],
            "example_photo": photo_path(s.photo_id).relative_to(_paths.DATA).as_posix(),
            "example_piece_corners_xy_m": r6(piece.outer),
            "example_frame_corners_xy_m": r6(piece.frame),
            "example_picture_corners_xy_m": r6(piece.picture),
        })
    (map_cfg.out_dir / "slots.json").write_text(json.dumps(slot_doc, ensure_ascii=False, indent=1),
                                                encoding="utf-8")


def halo(width=2.5):
    return [patheffects.withStroke(linewidth=width, foreground="white")]


def save_layout_guide(map_cfg, tag_cfg, slot_cfg, features, tags, slots, color_map: np.ndarray) -> None:
    s = map_cfg.size_m
    fig = plt.figure(figsize=(18, 19.2), dpi=200, facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
    ax.imshow(cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB), extent=[0, s, 0, s], interpolation="lanczos")
    for t in tags:
        color = "#d4002a" if t.kind == "building" else "#e07000"
        ax.add_patch(PolygonPatch(t.corners(tag_cfg.paper_mm / 1000), closed=True, fill=False, ec=color, lw=1.2))
        ax.text(*(t.center + np.array([0, tag_cfg.paper_mm / 1000 * 0.75])), str(t.id), color=color,
                fontsize=9, weight="bold", ha="center", va="bottom", path_effects=halo())
    for sl in slots:
        ax.add_patch(PolygonPatch(sl.corners(slot_cfg), closed=True, fill=False, ec="#0050d4", lw=1.8))
        top_mid = sl.corners(slot_cfg)[:2].mean(0) + sl.up * 0.012
        ax.text(*top_mid, f"S{sl.number}", color="#0050d4", fontsize=12, weight="bold", ha="center",
                va="center", path_effects=halo(3))
    ax.set_xlim(0, s)
    ax.set_ylim(0, s)
    ax.set_xticks(np.arange(0, s + 1e-9, 0.25))
    ax.set_yticks(np.arange(0, s + 1e-9, 0.25))
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.15)
    fig.text(0.02, 0.975, "Pasting plan: AprilTags and photo slots on the 2.5 m x 2.5 m color map",
             fontsize=20, weight="bold", va="top")
    n_building = sum(t.kind == "building" for t in tags)
    fig.text(0.02, 0.952,
             f"AprilTags {tag_cfg.family}, black square {tag_cfg.size_mm:g} mm, paper {tag_cfg.paper_mm:g} mm; "
             f"number = tag id.  Red: {n_building} in building corners.  Orange: {len(tags) - n_building} in corners "
             f"between roads.  Blue: {len(slots)} photo slots {slot_cfg.width_mm:g} x {slot_cfg.height_mm:g} mm.",
             fontsize=12, va="top")
    fig.text(0.02, 0.937, "Axes in meters from the lower-left corner of the map; north is up. "
             "Each tag's print page has a close-up picture of where it goes.", fontsize=12, va="top",
             color="#444")
    fig.savefig(map_cfg.out_dir / "layout_guide.png", dpi=200)
    plt.close(fig)


def crop_around(img: np.ndarray, center_xy, half_m: float, map_cfg: MapConfig) -> np.ndarray:
    """Square RGB crop of a PX_PER_MM render centered on a map point; outside the map is white."""
    c = to_px(center_xy, map_cfg)
    r = round(half_m * 1000 * PX_PER_MM)
    x, y = round(c[0]), round(c[1])
    padded = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.cvtColor(padded[y:y + 2 * r, x:x + 2 * r], cv2.COLOR_BGR2RGB)


def draw_guide(ax, img_rgb, center_xy, half_m, box_mm: tuple[float, float, float], outlines, caption: str):
    """Close-up of the map on a page: box_mm = (left, bottom, size) in page mm; outlines in map meters."""
    left, bottom, size = box_mm
    ax.imshow(img_rgb, extent=[left, left + size, bottom, bottom + size], interpolation="lanczos")
    to_page = lambda xy: np.column_stack([left + (xy[:, 0] - center_xy[0] + half_m) / (2 * half_m) * size,
                                          bottom + (xy[:, 1] - center_xy[1] + half_m) / (2 * half_m) * size])
    for quad, color in outlines:
        ax.add_patch(PolygonPatch(to_page(np.asarray(quad)), closed=True, fill=False, ec=color, lw=1.6))
    ax.add_patch(Rectangle((left, bottom), size, size, fill=False, ec="#333", lw=0.6))
    ax.add_patch(FancyArrow(left + size - 8, bottom + size - 20, 0, 12, width=1.6, head_width=5, head_length=4,
                            color="#222", length_includes_head=True))
    ax.text(left + size - 8, bottom + size - 25, "N", ha="center", va="center", fontsize=11, weight="bold",
            path_effects=halo())
    ax.text(left, bottom - 5, caption, fontsize=9, va="top", color="#333")


def scale_bar(ax, x, y):
    ax.add_patch(Rectangle((x, y), 50, 2.0, color="black", lw=0))
    for k in range(6):
        ax.plot([x + 10 * k] * 2, [y, y + 4], color="black", lw=0.6)
    ax.text(x + 52, y + 1, "50 mm: check with a ruler after printing", fontsize=8, va="center")


def new_page():
    fig = plt.figure(figsize=(A4_MM[0] / 25.4, A4_MM[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_MM[0])
    ax.set_ylim(0, A4_MM[1])
    ax.axis("off")
    return fig, ax  # one data unit is one millimeter on the page


def tag_page(t: TagPlacement, map_cfg, tag_cfg, features, guide_img):
    fig, ax = new_page()
    paper = tag_cfg.paper_mm
    ax.text(15, 284, f"AprilTag {tag_cfg.family}  ·  ID {t.id}", fontsize=22, weight="bold")
    ax.text(15, 276, place_label(t, features), fontsize=11)
    ax.text(15, 270, f"Black square {tag_cfg.size_mm:g} mm  ·  paper (cut size) {paper:g} mm  ·  "
            f"center ({t.center[0]:.4f}, {t.center[1]:.4f}) m  ·  yaw {t.yaw_deg:.2f}°", fontsize=9, color="#444")
    x0, y0 = (A4_MM[0] - paper) / 2, 255 - paper
    ax.imshow(tag_image(t.id, tag_cfg, cell_px=120), cmap="gray", vmin=0, vmax=255,
              extent=[x0, x0 + paper, y0, y0 + paper], interpolation="none")
    ax.add_patch(Rectangle((x0, y0), paper, paper, fill=False, ec="#a0a0a0", lw=0.3))
    ax.text(x0 + paper + 3, y0 + paper / 2, "← cut on the\n    gray line", fontsize=8, va="center", color="#666")
    half = 0.15
    guide = crop_around(guide_img, t.center, half, map_cfg)
    draw_guide(ax, guide, t.center, half, (45, 75, 120), [(t.corners(paper / 1000), "#d4002a")],
               "Where it goes: 30 x 30 cm of the map around the tag, north up. Red box = this paper.")
    if t.kind == "building":
        step3 = [f"3. Push the paper into the {t.corner_name} corner of the building, as in the picture, with its two",
                 "    edges just inside the building's dark outline (the outline stays visible)."]
    else:
        step3 = ["3. Lay one paper edge along the longer printed edge in the picture (gray road edge or dark",
                 "    building outline), slide it into the corner until it touches the other edge, and stick it there."]
    lines = [
        "1. Print at 100 % (\"Actual size\", not \"Fit to page\"), then check the bar below is 50 mm.",
        "2. Cut along the thin gray line.",
        *step3,
        "    Turn it so the pattern looks like the picture (north up). Never cover another tag or a photo.",
    ]
    ax.text(15, 60, "\n".join(lines), fontsize=9, va="top", linespacing=1.5)
    scale_bar(ax, 15, 20)
    return fig


def slot_page(s: Slot, map_cfg, slot_cfg, features, guide_img):
    fig, ax = new_page()
    path = photo_path(s.photo_id)
    img = read_image(path, cv2.IMREAD_COLOR)
    piece = photo_piece(s, img.shape[:2], slot_cfg)
    (pw, ph), (iw, ih) = piece.size_mm, piece.picture_mm
    f, m, bar = slot_cfg.frame_mm, slot_cfg.white_margin_mm, slot_cfg.bottom_bar_mm
    b = features["buildings"][s.building]
    ax.text(15, 284, f"Photo slot S{s.number}  ·  example photo {path.name}", fontsize=20, weight="bold")
    ax.text(15, 277, f"{building_label(b)}  ·  slot {slot_cfg.width_mm:g} x {slot_cfg.height_mm:g} mm  ·  "
            f"piece {pw:.1f} x {ph:.1f} mm", fontsize=10)
    ax.text(15, 271, f"picture, {f:g} mm black frame ({bar:g} mm under the picture), {m:g} mm white margin",
            fontsize=9, color="#444")
    x0, y0 = (A4_MM[0] - pw) / 2, 262 - ph  # page y points up: y0 is the piece's bottom edge
    ax.add_patch(Rectangle((x0 + m, y0 + m), pw - 2 * m, ph - 2 * m, color="black", lw=0, zorder=1))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
              extent=[x0 + m + f, x0 + m + f + iw, y0 + m + bar, y0 + m + bar + ih], interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((x0, y0), pw, ph, fill=False, ec="#a0a0a0", lw=0.3, zorder=3))
    ax.text(x0 + pw / 2, y0 + ph + 2, "▲ top", ha="center", fontsize=8, color="#666")
    half = 0.2
    guide = crop_around(guide_img, s.center, half, map_cfg)
    draw_guide(ax, guide, s.center, half, (55, 50, 100),
               [(s.corners(slot_cfg), "#0050d4"), (piece.outer, "#d4002a")],
               "Where it goes: 40 x 40 cm of the map, north up. Blue = slot, red = this piece.")
    lines = [
        "1. Print at 100 % (\"Actual size\"), then check the bar below is 50 mm.",
        "2. Cut along the thin gray line, keeping the white margin around the black frame.",
        "3. Paste it centered in the slot as in the picture, top (▲) toward the slot's north-facing side.",
        "   Any test photo can go into any slot; print new ones the same way (frame, thick bar under the",
        "   picture, white margin): the camera uses the bar to turn the photo upright for the LLM.",
    ]
    ax.text(15, 40, "\n".join(lines), fontsize=9, va="top", linespacing=1.4)
    scale_bar(ax, 15, 8)
    return fig


def save_prints(map_cfg, tag_cfg, slot_cfg, features, tags, slots, guide_img) -> None:
    tag_dir = map_cfg.print_dir / "tags"
    photo_dir = map_cfg.print_dir / "photos"
    for d in (tag_dir, photo_dir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.pdf"):
            old.unlink()
    with PdfPages(map_cfg.print_dir / "all_tags.pdf") as combined:
        for t in tags:
            fig = tag_page(t, map_cfg, tag_cfg, features, guide_img)
            fig.savefig(tag_dir / f"tag_{t.id:02d}.pdf")
            combined.savefig(fig)
            plt.close(fig)
    for s in slots:
        fig = slot_page(s, map_cfg, slot_cfg, features, guide_img)
        fig.savefig(photo_dir / f"slot_{s.number}.pdf")
        plt.close(fig)


def build(map_cfg=MapConfig(), tag_cfg=TagConfig(), slot_cfg=SlotConfig()):
    """Choose slots, building tags, then road tags. Returns (features, slots, tags, stats)."""
    features, polys = load_buildings(map_cfg)
    gate_xy = features["landmarks"]["main_gate"]["xy_m"]
    slots = choose_slots(polys, map_cfg, slot_cfg, gate_xy)
    tags, n_spots = choose_tags(polys, map_cfg, tag_cfg, slots, slot_cfg, gate_xy)
    stats = {"building_spots": n_spots, "building_tags": len(tags)}
    if tag_cfg.road_tags:
        road, road_stats = choose_road_tags(features, polys, map_cfg, tag_cfg, slots, slot_cfg, gate_xy, tags)
        tags = tags + road
        stats |= road_stats | {"road_tags": len(road)}
    return features, slots, tags, stats


def main():
    map_cfg, tag_cfg, slot_cfg = MapConfig(), TagConfig(), SlotConfig()
    map_cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["DejaVu Sans"] + [f for f in KOREAN_FONTS
                                                     if f in set(font_manager.get_font_names())][:1]
    plt.rcParams["pdf.fonttype"] = 42
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # no bold Korean face: regular is fine

    features, slots, tags, stats = build(map_cfg, tag_cfg, slot_cfg)
    write_json(map_cfg, tag_cfg, slot_cfg, features, tags, slots)

    with_tags = render_finished_map(map_cfg, tag_cfg, slot_cfg, tags, slots, gray=False, with_photos=False)
    write_image(map_cfg.out_dir / "map_with_tags.png", with_tags)
    write_image(map_cfg.out_dir / "map_with_tags_gray.png", cv2.cvtColor(with_tags, cv2.COLOR_BGR2GRAY))
    with_photos = with_tags.copy()
    render_photos(with_photos, slots, map_cfg, slot_cfg)
    write_image(map_cfg.out_dir / "map_with_photos.png", with_photos)
    write_image(map_cfg.out_dir / "map_with_photos_gray.png", cv2.cvtColor(with_photos, cv2.COLOR_BGR2GRAY))

    save_layout_guide(map_cfg, tag_cfg, slot_cfg, features, tags, slots, with_photos)
    save_prints(map_cfg, tag_cfg, slot_cfg, features, tags, slots, with_photos)

    print(f"{stats['building_tags']} building tags from {stats['building_spots']} square building corners "
          f"that fit a {tag_cfg.paper_mm:g} mm paper")
    if tag_cfg.road_tags:
        print(f"{stats['road_tags']} road tags from {stats['road_spots']} corners between roads; road length "
              f"within {tag_cfg.road_cover_radius_m} m of a tag: {stats['road_covered_before']:.0%} -> "
              f"{stats['road_covered_after']:.0%}")
    for s in slots:
        print(f"  slot S{s.number}: {building_label(features['buildings'][s.building])} at "
              f"({s.center[0]:.3f}, {s.center[1]:.3f}) m, {s.cover:.0%} on the building, "
              f"example photo {s.photo_id}")
    print(f"Wrote {map_cfg.out_dir} and {map_cfg.print_dir}")


if __name__ == "__main__":
    main()
