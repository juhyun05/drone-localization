# KAIST 2.5 × 2.5 m 모형 기준 지도

원본 OSM을 모형 좌표로 변환해 위치 추정과 인쇄에 사용할 세 파일을 만든다. [원본 OSM](../data/gis/KAIST_map.osm) · [생성기](build_campus_map.py)

| 산출물 | 내용 |
| --- | --- |
| [모형 지물 JSON](../data/maps/KAIST_model_features_2p5m.json) | 건물 60개의 외곽선·색상, 도로 46개의 중심선, 정문 출입구 4개와 평균 좌표 |
| [위치 정합용 기준 이미지](../data/maps/KAIST_model_reference_2p5m.png) | 정문을 빨간 원으로 표시한 2000 × 2000 px 지도 |
| [설명용 지도](../data/maps/KAIST_model_map_2p5m_roads_gate.png) | 정문 위치, 도로명, 눈금과 범례 |

## 실행

저장소 루트에서 실행하면 세 파일을 `data/maps/`에 쓴다.

```powershell
python -m pip install -r map_tools/requirements.txt
python map_tools/build_campus_map.py
```

다른 출력 위치를 지정하려면 `python map_tools/build_campus_map.py --input data/gis/KAIST_map.osm --output-dir outputs/map_check`를 실행한다. 다른 Python 모듈에서도 `map_tools.build_campus_map.build_campus_map(input_osm, output_dir)`를 호출할 수 있다. 설명용 지도는 설치된 나눔고딕, 맑은 고딕, Apple SD Gothic Neo, Noto Sans CJK KR 순으로 한글 글꼴을 선택한다. 출력용 EPS·PDF·PNG 제작은 [실사출력 안내](README.md)를 따른다.

AprilTag 배치 파일은 기준 지물 JSON의 SHA-256으로 인쇄 지도와의 일치를 확인한다. 생성기는 모든 지물·원본 좌표·메타데이터가 배포본과 같고 좌표 변환 라이브러리의 부동소수점 차이만 있을 때 기존 축척 값을 유지한다. 배포본이 없거나 지물이 바뀌면 새로 계산한 값을 기록한다. JSON은 모든 운영체제에서 UTF-8과 LF 줄바꿈으로 저장한다.

## 지도 좌표계 및 축척

- 우측이 +x(동쪽), 상단이 +y(북쪽), 좌하단 원점 (0, 0)
- `0 ≤ x ≤ 2.5m`, `0 ≤ y ≤ 2.5m`
- 원본 WGS84(위도/경도) → UTM 52N(EPSG:32652) → 동일 축척의 2.5×2.5m 모형

## 반영한 OSM 특성

- 건물: `building=*`
- 큰 도로: `highway=secondary`
- 교내 주요 도로: `highway=unclassified`
- 연결·진입 도로: `highway=service`, 단 `service=parking_aisle` 제외
- 정문: `barrier=entrance`, `name=KAIST 정문`인 OSM 노드 4개의 평균 위치

```python
import json
from pathlib import Path
path = Path('data/maps/KAIST_model_features_2p5m.json')
features = json.loads(path.read_text(encoding='utf-8'))
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
