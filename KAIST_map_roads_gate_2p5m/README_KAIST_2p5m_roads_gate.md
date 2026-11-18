# KAIST 2.5×2.5m 모형 지도: 건물 + 주요 도로 + 정문

## 결과물
- `KAIST_model_map_2p5m_roads_gate.png`: 정문 위치, 도로명, 범례가 있는 **사람이 보는 지도**
- `KAIST_model_reference_2p5m.png`: 제목/좌표/범례가 없는 **위치 정합용 기준 래스터** (2000×2000px)
- `KAIST_model_features_2p5m.json`: 기존 건물 60개 + 도로 46개(지도 중심선) + 정문 출입구 원본 노드 4개
- `osm_map_roads_gate_2p5m.py`: OSM을 읽어 모든 결과물을 재생성하는 Python 코드
- `KAIST_map.osm`: 원본 OSM 파일

## 실행
```bash
pip install numpy matplotlib shapely pyproj
python osm_map_roads_gate_2p5m.py
```
한글 라벨은 실행 환경에 `/usr/share/fonts/truetype/nanum/NanumGothic.ttf`가 있으면 적용되고, 없는 환경에서는 Matplotlib이 선택한 폰트에 따라 글자가 깨질 수 있습니다. Windows에서는 시스템의 한글 지원 글꼴로 변경하세요.

## 지도 좌표계 및 축척
- 우측이 +x(동쪽), 상단이 +y(북쪽), 좌하단 원점 (0, 0)
- `0 ≤ x ≤ 2.5m`, `0 ≤ y ≤ 2.5m`
- 원본 WGS84(위도/경도) → UTM 52N(EPSG:32652) → 동일 축척의 2.5×2.5m 모형
- 기존 건물 좌표, 외곽선, 건물별 색상은 `KAIST_model_buildings_2p5m.json`과 동일

## 반영한 OSM 특성
- 건물: `building=*`
- 큰 도로: `highway=secondary`
- 교내 주요 도로: `highway=unclassified`
- 연결·진입 도로: `highway=service`, 단 `service=parking_aisle` 제외
- 정문: `barrier=entrance`, `name=KAIST 정문`인 OSM 노드 4개의 평균 위치

```python
import json
from pathlib import Path
features = json.loads(Path('KAIST_model_features_2p5m.json').read_text(encoding='utf-8'))
main_gate_xy = features['landmarks']['main_gate']['xy_m']  # [1.53157, 0.70182]
roads = features['roads']
buildings = features['buildings']
```

## 주의 사항
1. **도로는 OSM 중심선 데이터**입니다. 실제 차도 폭은 포함되어 있지 않으며, 그림의 선 굵기는 가독성을 위해 정한 값입니다. 3D 모형을 제작할 때는 실제 적용할 차도 폭으로 따로 정해야 합니다.
2. **건물별 색상은 임의 배정한 식별 색상**이며 실제 KAIST 건물 지붕의 색상이 아닙니다. 카메라 기반 색상 정합을 하려면 실제 제작하는 모형에도 색상을 맞춰야 합니다.
3. 메인 게이트 마커는 출입구 노드 4개의 평균 좌표에 표시했습니다. 따라서 개별 차선·게이트 구조의 정확한 측량 좌표가 아닙니다.
4. 위치 추정용 기준 이미지에는 텍스트·눈금이 없지만 정문을 빨간 원으로 표기했습니다. 실물 모형에서 동일 마커를 사용하지 않을 경우 정합 알고리즘에서 마스킹하거나 해당 표식을 제거하세요.

원본 지도 데이터: © OpenStreetMap contributors
