"""Build the 2.5 m KAIST model geometry and maps from the campus OSM file."""

import argparse
import colorsys
import json
import xml.etree.ElementTree as ET
from collections import Counter
from functools import cache
from math import isclose
from pathlib import Path

from matplotlib import font_manager, patheffects, rc_context
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon as PolygonPatch, Rectangle
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, box

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
DEFAULT_INPUT = DATA_DIR / 'gis' / 'KAIST_map.osm'
DEFAULT_OUTPUT = DATA_DIR / 'maps'
PREVIEW_NAME = 'KAIST_model_map_2p5m_roads_gate.png'
REFERENCE_NAME = 'KAIST_model_reference_2p5m.png'
FEATURES_NAME = 'KAIST_model_features_2p5m.json'
MODEL_SIZE_M = 2.5

# Road display specification, drawn under buildings.
# Values represent line thickness on the drawing (not surveyed road width).
ROAD_STYLES = {
    'secondary': {'color': '#6f7781', 'outline': '#fdfdfd', 'width_pt': 10.5},
    'unclassified': {'color': '#858f98', 'outline': '#fdfdfd', 'width_pt': 7.5},
    'service': {'color': '#c3cad1', 'outline': '#ffffff', 'width_pt': 3.8},
}
ROAD_NAME_LABELS = [
    ('대학로', (2.04, 0.91), -28),
    ('카이서로', (0.93, 0.47), 21),
    ('대학로291번길', (0.64, 1.73), 60),
    ('대학로291번나길', (2.14, 1.58), 55),
    ('대학로291번다길', (1.89, 2.28), -22),
]

# The OSM main gate is represented by four entrance nodes; their mean is the
# approximate symbol position, while all exact nodes are retained in the JSON.
MAIN_GATE_NAME = 'KAIST 정문'


def get_tags(elem):
    return {t.get('k'): t.get('v') for t in elem.findall('tag')}


def polygon_parts(p):
    if p.geom_type == 'Polygon':
        return [p]
    return [g for g in getattr(p, 'geoms', []) if g.geom_type == 'Polygon']


def line_parts(g):
    if g.geom_type == 'LineString':
        return [g]
    return [sub for sub in getattr(g, 'geoms', []) if sub.geom_type == 'LineString']


def model_coordinates(root, node_ll):
    to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32652', always_xy=True)
    bounds = root.find('bounds')
    lat0, lat1 = float(bounds.get('minlat')), float(bounds.get('maxlat'))
    lon0, lon1 = float(bounds.get('minlon')), float(bounds.get('maxlon'))
    corners = [to_utm.transform(lo, la) for lo in (lon0, lon1) for la in (lat0, lat1)]
    min_e, max_e = min(p[0] for p in corners), max(p[0] for p in corners)
    min_n, max_n = min(p[1] for p in corners), max(p[1] for p in corners)
    real_w, real_h = max_e - min_e, max_n - min_n
    scale = MODEL_SIZE_M / max(real_w, real_h)
    offset_x = (MODEL_SIZE_M - scale * real_w) / 2
    offset_y = (MODEL_SIZE_M - scale * real_h) / 2

    @cache
    def node_xy(ref):
        lon, lat = node_ll[ref]
        e, n = to_utm.transform(lon, lat)
        return (round(offset_x + (e - min_e) * scale, 6),
                round(offset_y + (n - min_n) * scale, 6))

    return node_xy, real_w, real_h, scale


def polygon_from_way(w, node_ll, node_xy):
    refs = [nd.get('ref') for nd in w.findall('nd')]
    if len(refs) < 4 or refs[0] != refs[-1] or any(r not in node_ll for r in refs):
        return None
    p = Polygon([node_xy(r) for r in refs])
    if not p.is_valid:
        p = p.buffer(0)
    return p if not p.is_empty else None


def collect_buildings(ways, relations, node_ll, node_xy, extent):
    # 1. Buildings: preserve original golden-angle color generation order.
    candidates = []
    for way_id, w in ways.items():
        tags = get_tags(w)
        if tags.get('building', 'no') == 'no':
            continue
        geom = polygon_from_way(w, node_ll, node_xy)
        if geom is not None:
            candidates.append((f'way/{way_id}', tags, geom))

    for relation in relations:
        tags = get_tags(relation)
        if tags.get('building', 'no') == 'no' or tags.get('type') != 'multipolygon':
            continue
        for member in relation.findall('member'):
            if member.get('type') != 'way' or member.get('role') != 'outer':
                continue
            w = ways.get(member.get('ref'))
            if w is None:
                continue
            geom = polygon_from_way(w, node_ll, node_xy)
            if geom is not None:
                candidates.append((f"relation/{relation.get('id')}", tags, geom))

    buildings = []
    for key, tags, geom in candidates:
        clipped = geom.intersection(extent)
        if clipped.is_empty:
            continue
        parts = polygon_parts(clipped)
        for part_num, part in enumerate(parts):
            if part.area <= 1e-8 * (MODEL_SIZE_M / 5.0) ** 2:
                continue
            idx = len(buildings)
            hue = (idx * 0.61803398875) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.58, 0.95)
            color = '#%02x%02x%02x' % tuple(round(255 * a) for a in rgb)
            center = part.centroid
            buildings.append({
                'id': key if len(parts) == 1 else f'{key}/{part_num}',
                'name': tags.get('name', ''),
                'ref': tags.get('ref', ''),
                'color': color,
                'polygon_xy_m': [[round(x, 5), round(y, 5)] for x, y in part.exterior.coords],
                'centroid_xy_m': [round(center.x, 5), round(center.y, 5)],
            })
    return buildings


def collect_roads(ways, node_ll, node_xy, extent):
    # 2. Major roads, based on OSM highway category. Do not show parking aisles,
    # pedestrian paths, crossings or footways in a major-road map.
    roads = []
    for way_id, w in ways.items():
        tags = get_tags(w)
        highway = tags.get('highway')
        if highway not in ROAD_STYLES:
            continue
        if highway == 'service' and tags.get('service') == 'parking_aisle':
            continue
        refs = [nd.get('ref') for nd in w.findall('nd')]
        if len(refs) < 2 or any(r not in node_ll for r in refs):
            continue
        line = LineString([node_xy(ref) for ref in refs])
        clipped = line.intersection(extent)
        for partnum, part in enumerate(line_parts(clipped)):
            if part.length < 0.002:
                continue
            roads.append({
                'id': f'way/{way_id}' if partnum == 0 else f'way/{way_id}/{partnum}',
                'name': tags.get('name', ''),
                'highway': highway,
                'centerline_xy_m': [[round(x, 5), round(y, 5)] for x, y in part.coords],
                'display_color': ROAD_STYLES[highway]['color'],
                'display_linewidth_pt': ROAD_STYLES[highway]['width_pt'],
            })
    return roads


def collect_main_gate(nodes, node_ll, node_xy):
    # 3. Main gate. Keep the source point positions as well as their mean.
    gate_nodes = []
    for node in nodes:
        tags = get_tags(node)
        if tags.get('name') == MAIN_GATE_NAME and tags.get('barrier') == 'entrance':
            lon, lat = node_ll[node.get('id')]
            x, y = node_xy(node.get('id'))
            if 0 <= x <= MODEL_SIZE_M and 0 <= y <= MODEL_SIZE_M:
                gate_nodes.append({'id': 'node/' + node.get('id'),
                                   'lon': lon, 'lat': lat,
                                   'xy_m': [round(x, 5), round(y, 5)]})
    if not gate_nodes:
        raise ValueError('No OSM nodes with name=KAIST 정문, barrier=entrance were found')
    gate_centroid = [round(sum(p['xy_m'][j] for p in gate_nodes) / len(gate_nodes), 5)
                     for j in (0, 1)]
    return gate_nodes, gate_centroid


def read_map_features(input_osm: Path) -> dict:
    root = ET.parse(input_osm).getroot()
    nodes = root.findall('node')
    node_ll = {n.get('id'): (float(n.get('lon')), float(n.get('lat'))) for n in nodes}
    ways = {w.get('id'): w for w in root.findall('way')}
    node_xy, real_w, real_h, scale = model_coordinates(root, node_ll)
    extent = box(0, 0, MODEL_SIZE_M, MODEL_SIZE_M)
    buildings = collect_buildings(ways, root.findall('relation'), node_ll, node_xy, extent)
    roads = collect_roads(ways, node_ll, node_xy, extent)
    gate_nodes, gate_centroid = collect_main_gate(nodes, node_ll, node_xy)

    return {
        'metadata': {
            'source': input_osm.name,
            'attribution': '© OpenStreetMap contributors',
            'coordinate_frame': 'model meters; lower-left origin; +x east, +y north',
            'model_width_m': MODEL_SIZE_M,
            'model_height_m': MODEL_SIZE_M,
            'real_bounds_width_m': round(real_w, 2),
            'real_bounds_height_m': round(real_h, 2),
            'scale_model_per_real': scale,
            'road_selection': 'highway=secondary/unclassified/service, excluding service=parking_aisle; highway tags from source OSM',
            'roads_are_centerlines': True,
            'notes': 'Building colors, road colors, gate indicator and drawn road widths are visualization choices. Match the physical model when using color-based localization.'
        },
        'buildings': buildings,
        'roads': roads,
        'landmarks': {
            'main_gate': {
                'name': MAIN_GATE_NAME,
                'xy_m': gate_centroid,
                'source_nodes': gate_nodes,
                'interpretation': 'Mean of four OSM entrance nodes named KAIST 정문 (symbol position, not a surveyed control point)'
            }
        }
    }


def draw_map(ax, data, *, annotated):
    ax.set_facecolor('#edf1e9')
    for highway in ('service', 'unclassified', 'secondary'):
        sty = ROAD_STYLES[highway]
        for road in data['roads']:
            if road['highway'] != highway:
                continue
            xs = [p[0] for p in road['centerline_xy_m']]
            ys = [p[1] for p in road['centerline_xy_m']]
            ax.plot(xs, ys, color=sty['outline'], linewidth=sty['width_pt'] + 2.2,
                    zorder=2, solid_joinstyle='round', solid_capstyle='round')
            ax.plot(xs, ys, color=sty['color'], linewidth=sty['width_pt'],
                    zorder=3, solid_joinstyle='round', solid_capstyle='round')

    for building in data['buildings']:
        ax.add_patch(PolygonPatch(building['polygon_xy_m'], closed=True,
                                  facecolor=building['color'], edgecolor='#283240',
                                  linewidth=0.7 if annotated else 0.55, zorder=5))
    gate_x, gate_y = data['landmarks']['main_gate']['xy_m']
    if annotated:
        ax.scatter([gate_x], [gate_y], marker='*', s=265, c='#dd343f',
                   edgecolors='white', linewidths=1.3, zorder=12)
        ax.annotate('KAIST 정문', xy=(gate_x, gate_y), xytext=(1.0, 0.31),
                    fontsize=12, weight='bold', color='#b21e2b',
                    bbox={'boxstyle': 'round,pad=0.32', 'fc': 'white', 'ec': '#d86a73', 'lw': 1},
                    arrowprops={'arrowstyle': '-|>', 'color': '#c72935', 'lw': 1.8,
                                'shrinkA': 8, 'shrinkB': 8, 'connectionstyle': 'arc3,rad=0.15'},
                    zorder=13)
        # Label only named major arteries. For all labels, use a white halo.
        for name, (x, y), angle in ROAD_NAME_LABELS:
            txt = ax.text(x, y, name, fontsize=8.5, color='#202c33', rotation=angle,
                          rotation_mode='anchor', weight='bold', ha='center', va='center',
                          zorder=10)
            txt.set_path_effects([patheffects.withStroke(linewidth=2.4, foreground='white')])
        ax.text(0.055, 2.39, 'N ↑', fontsize=12, weight='bold', color='#263b48', zorder=11)
    else:
        # The reference raster represents the main gate as a reproducible red marker.
        ax.add_patch(Circle((gate_x, gate_y), radius=0.016, facecolor='#dd343f',
                            edgecolor='white', linewidth=0.7, zorder=12))
    ax.set_xlim(0, MODEL_SIZE_M)
    ax.set_ylim(0, MODEL_SIZE_M)
    ax.set_aspect('equal', adjustable='box')
    if annotated:
        ticks = [i * MODEL_SIZE_M / 10 for i in range(11)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(alpha=0.12, zorder=0)
        ax.set_xlabel('모형 x (m) · 동쪽')
        ax.set_ylabel('모형 y (m) · 북쪽')
    else:
        ax.set_axis_off()


def write_annotated_map(data, output_path):
    # Preview: annotated map + compact legend in a non-map side panel.
    fig = Figure(figsize=(12.4, 10.7), dpi=170, facecolor='white')
    ax = fig.add_axes([0.067, 0.10, 0.73, 0.80])
    draw_map(ax, data, annotated=True)
    fig.suptitle('KAIST 모형 지도 | 건물 · 주요 도로 · 정문', fontsize=18, weight='bold', y=0.971)
    legend_ax = fig.add_axes([0.81, 0.20, 0.17, 0.64])
    legend_ax.axis('off')
    legend_ax.text(0, 1.00, '범례', fontsize=14, weight='bold', transform=legend_ax.transAxes)
    legend_items = [
        ('건물 (기존 색상)', '#e46f89', None),
        ('큰 도로 · 대학로', ROAD_STYLES['secondary']['color'], 'secondary'),
        ('교내 주요 도로', ROAD_STYLES['unclassified']['color'], 'unclassified'),
        ('진입 · 연결 도로', ROAD_STYLES['service']['color'], 'service'),
        ('학교 정문', '#dd343f', 'gate'),
    ]
    for i, (name, col, kind) in enumerate(legend_items):
        ypos = .90-i*.125
        if kind == 'gate':
            legend_ax.scatter([0.11], [ypos], marker='*', c=col, s=200, edgecolor='white',
                              linewidth=.8, transform=legend_ax.transAxes)
        elif kind is None:
            legend_ax.add_patch(Rectangle((.025, ypos-.019), .16, .040, color=col,
                                                ec='#283240', transform=legend_ax.transAxes))
        else:
            legend_ax.plot([.02,.19], [ypos,ypos], transform=legend_ax.transAxes,
                           lw=ROAD_STYLES[kind]['width_pt'], color=col, solid_capstyle='round')
        legend_ax.text(.24, ypos, name, va='center', fontsize=9.8, transform=legend_ax.transAxes)
    legend_ax.text(0, .17, 'OSM 도로는 중심선으로 제공됨\n표시 폭은 시각화용임', fontsize=8.5,
                   color='#53636c', va='top', transform=legend_ax.transAxes)
    fig.text(.48, .029,
             '자료: © OpenStreetMap contributors  |  좌표: 2.5 × 2.5 m (좌하단 원점, 동쪽 +x, 북쪽 +y)',
             ha='center', fontsize=10, color='#4f5a61')
    fig.savefig(output_path, dpi=170)


def write_reference_map(data, output_path):
    # Image without annotations; aspect exactly 1:1, no axes or margins.
    fig = Figure(figsize=(10,10), dpi=200, facecolor='white')
    ax = fig.add_axes([0,0,1,1])
    draw_map(ax, data, annotated=False)
    fig.savefig(output_path, dpi=200, pad_inches=0)


def preserve_reference_scale(data: dict, reference_path: Path) -> None:
    """Keep the printed map's checksum when only PROJ's last scale digits differ."""
    try:
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        stored_scale = reference['metadata']['scale_model_per_real']
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return
    scale = data['metadata']['scale_model_per_real']
    if not isinstance(stored_scale, (int, float)) or not isclose(stored_scale, scale, rel_tol=1e-12):
        return
    reference['metadata']['scale_model_per_real'] = scale
    if reference == data:
        data['metadata']['scale_model_per_real'] = stored_scale


def build_campus_map(input_osm: Path, output_dir: Path) -> dict:
    """Write the geometry JSON, labeled preview and localization raster to output_dir."""
    data = read_map_features(Path(input_osm))
    preserve_reference_scale(data, DEFAULT_OUTPUT / FEATURES_NAME)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / FEATURES_NAME).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n'
    )
    installed = set(font_manager.get_font_names())
    options = {'axes.unicode_minus': False}
    for family in ('NanumGothic', 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans CJK KR'):
        if family in installed:
            options['font.family'] = family
            break
    with rc_context(options):
        write_annotated_map(data, output_dir / PREVIEW_NAME)
        write_reference_map(data, output_dir / REFERENCE_NAME)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='source OSM XML')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = build_campus_map(args.input, args.output_dir)
    roads = data['roads']
    gate = data['landmarks']['main_gate']
    print('Buildings:', len(data['buildings']))
    print('Roads:', len(roads), dict(Counter(r['highway'] for r in roads)))
    print('Main gate:', gate['xy_m'], '(from', len(gate['source_nodes']), 'OSM nodes)')
    for name in (PREVIEW_NAME, REFERENCE_NAME, FEATURES_NAME):
        print(args.output_dir / name)


if __name__ == '__main__':
    main()
