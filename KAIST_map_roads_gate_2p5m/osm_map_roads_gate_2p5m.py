"""KAIST OSM 2.5m model map, with buildings, major roads, and the main gate.

Usage:
    python -m pip install -r ../requirements.txt
    python osm_map_roads_gate_2p5m.py

Input: KAIST_map.osm (the source supplied by the user)
Output:
    KAIST_model_map_2p5m_roads_gate.png   human-readable map with labels
    KAIST_model_reference_2p5m.png        label-free image for localization
    KAIST_model_features_2p5m.json       buildings + roads + main-gate coordinates

The colors and widths of roads and buildings are display/model labels. OSM
records road centerlines, not surveyed road width; physical model needs to
apply the same artificial colors/geometry to be useful for image matching.
"""

import colorsys
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch, Circle
from matplotlib.lines import Line2D
from matplotlib import patheffects, font_manager
from pyproj import Transformer
from shapely.geometry import Polygon, LineString, box

HERE = Path(__file__).resolve().parent
INPUT_OSM = HERE / 'KAIST_map.osm'
PREVIEW_PNG = HERE / 'KAIST_model_map_2p5m_roads_gate.png'
REFERENCE_PNG = HERE / 'KAIST_model_reference_2p5m.png'
OUTPUT_JSON = HERE / 'KAIST_model_features_2p5m.json'
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

# OSMap main gate is represented by four entrance nodes; their mean is the
# approximate symbol position, while all exact nodes are retained in the JSON.
MAIN_GATE_NAME = 'KAIST 정문'


root = ET.parse(INPUT_OSM).getroot()
node_ll = {
    n.get('id'): (float(n.get('lon')), float(n.get('lat')))
    for n in root.findall('node')
}
ways = {w.get('id'): w for w in root.findall('way')}
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
extent = box(0, 0, MODEL_SIZE_M, MODEL_SIZE_M)


def model_xy(lon, lat):
    e, n = to_utm.transform(lon, lat)
    return (round(offset_x + (e - min_e) * scale, 6),
            round(offset_y + (n - min_n) * scale, 6))


def get_tags(elem):
    return {t.get('k'): t.get('v') for t in elem.findall('tag')}


def polygon_from_way(w):
    refs = [nd.get('ref') for nd in w.findall('nd')]
    if len(refs) < 4 or refs[0] != refs[-1] or any(r not in node_ll for r in refs):
        return None
    p = Polygon([model_xy(*node_ll[r]) for r in refs])
    if not p.is_valid:
        p = p.buffer(0)
    return p if not p.is_empty else None


def polygon_parts(p):
    if p.geom_type == 'Polygon':
        return [p]
    return [g for g in getattr(p, 'geoms', []) if g.geom_type == 'Polygon']


def line_parts(g):
    if g.geom_type == 'LineString':
        return [g]
    return [sub for sub in getattr(g, 'geoms', []) if sub.geom_type == 'LineString']


# 1. Buildings: preserve original golden-angle color generation order.
candidates = []
for way_id, w in ways.items():
    tags = get_tags(w)
    if tags.get('building', 'no') == 'no':
        continue
    geom = polygon_from_way(w)
    if geom is not None:
        candidates.append((f'way/{way_id}', tags, geom))

for relation in root.findall('relation'):
    tags = get_tags(relation)
    if tags.get('building', 'no') == 'no' or tags.get('type') != 'multipolygon':
        continue
    for member in relation.findall('member'):
        if member.get('type') != 'way' or member.get('role') != 'outer':
            continue
        w = ways.get(member.get('ref'))
        if w is None:
            continue
        geom = polygon_from_way(w)
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
        buildings.append({
            'id': key if len(parts) == 1 else f'{key}/{part_num}',
            'name': tags.get('name', ''),
            'ref': tags.get('ref', ''),
            'color': color,
            'polygon_xy_m': [[round(x, 5), round(y, 5)] for x, y in part.exterior.coords],
            'centroid_xy_m': [round(part.centroid.x, 5), round(part.centroid.y, 5)],
        })

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
    line = LineString([model_xy(*node_ll[ref]) for ref in refs])
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

# 3. Main gate. Keep the source point positions as well as their mean.
gate_nodes = []
for node in root.findall('node'):
    tags = get_tags(node)
    if tags.get('name') == MAIN_GATE_NAME and tags.get('barrier') == 'entrance':
        lon, lat = node_ll[node.get('id')]
        x, y = model_xy(lon, lat)
        if 0 <= x <= MODEL_SIZE_M and 0 <= y <= MODEL_SIZE_M:
            gate_nodes.append({'id': 'node/' + node.get('id'),
                               'lon': lon, 'lat': lat,
                               'xy_m': [round(x, 5), round(y, 5)]})
if not gate_nodes:
    raise ValueError('No OSM nodes with name=KAIST 정문, barrier=entrance were found')
gate_centroid = [round(sum(p['xy_m'][j] for p in gate_nodes) / len(gate_nodes), 5)
                 for j in (0, 1)]

out = {
    'metadata': {
        'source': INPUT_OSM.name,
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
OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

installed_fonts = set(font_manager.get_font_names())
for family in ('NanumGothic', 'Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans CJK KR'):
    if family in installed_fonts:
        plt.rcParams['font.family'] = family
        break
plt.rcParams['axes.unicode_minus'] = False


def draw_map(ax, *, annotated):
    ax.set_facecolor('#edf1e9')
    for highway in ('service', 'unclassified', 'secondary'):
        sty = ROAD_STYLES[highway]
        for road in roads:
            if road['highway'] != highway:
                continue
            xs = [p[0] for p in road['centerline_xy_m']]
            ys = [p[1] for p in road['centerline_xy_m']]
            ax.plot(xs, ys, color=sty['outline'], linewidth=sty['width_pt'] + 2.2,
                    zorder=2, solid_joinstyle='round', solid_capstyle='round')
            ax.plot(xs, ys, color=sty['color'], linewidth=sty['width_pt'],
                    zorder=3, solid_joinstyle='round', solid_capstyle='round')

    for building in buildings:
        ax.add_patch(PolygonPatch(building['polygon_xy_m'], closed=True,
                                  facecolor=building['color'], edgecolor='#283240',
                                  linewidth=0.7 if annotated else 0.55, zorder=5))
    gate_x, gate_y = gate_centroid
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


# Preview: annotated map + compact legend in a non-map side panel.
fig = plt.figure(figsize=(12.4, 10.7), dpi=170, facecolor='white')
ax = fig.add_axes([0.067, 0.10, 0.73, 0.80])
draw_map(ax, annotated=True)
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
        legend_ax.add_patch(plt.Rectangle((.025, ypos-.019), .16, .040, color=col,
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
fig.savefig(PREVIEW_PNG, dpi=170)
plt.close(fig)

# Image without annotations; aspect exactly 1:1, no axes or margins.
fig = plt.figure(figsize=(10,10), dpi=200, facecolor='white')
ax = fig.add_axes([0,0,1,1])
draw_map(ax, annotated=False)
fig.savefig(REFERENCE_PNG, dpi=200, pad_inches=0)
plt.close(fig)

print('Buildings:', len(buildings))
print('Roads:', len(roads), dict(Counter(r['highway'] for r in roads)))
print('Main gate:', gate_centroid, '(from', len(gate_nodes), 'OSM nodes)')
print('Output:', PREVIEW_PNG, REFERENCE_PNG, OUTPUT_JSON)
