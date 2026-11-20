# KAIST 캠퍼스 지도 제작과 실사출력

인쇄 업체에는 2,500 × 2,500 mm 크기의 한 장짜리 벡터 EPS를 전달한다. 제작 파일과 검토본은 [인쇄 폴더](../data/print/campus/)에 있다. [배포용 ZIP](../data/print/campus_bundle.zip)

| 파일 | 용도 |
| --- | --- |
| [KAIST_map_PRINT_2500x2500mm.eps](../data/print/campus/KAIST_map_PRINT_2500x2500mm.eps) | 출력 업체 제출, 실물 크기 벡터 |
| [KAIST_map_PRINT_preview.pdf](../data/print/campus/KAIST_map_PRINT_preview.pdf) | 실물 크기 벡터 검토본 |
| [KAIST_map_PRINT_preview.png](../data/print/campus/KAIST_map_PRINT_preview.png) | 화면 검토용 2000 × 2000px 이미지 |

PNG를 기준 시안으로 공유하고, 같은 지도의 EPS를 2,500 × 2,500mm 제작 파일로 사용한다. PNG를 그 크기로 배치하면 전체 해상도는 약 20.3ppi다.

## 생성 및 검증

Python 3.10 이상에서 저장소 루트를 기준으로 실행한다. OSM 원본에서 기준 지도까지 다시 만들려면 먼저 [기준 지도 제작 절차](map_source.md)를 따른다.

```powershell
python -m pip install -r map_tools/requirements.txt
python map_tools/create_print_map.py
python map_tools/verify_print_map.py
```

생성기는 [KAIST 지도 JSON](../data/maps/KAIST_model_features_2p5m.json)을 직접 읽는다. 세 출력물은 동일한 건물·도로·정문 벡터 경로에서 생성한다. 도로 선폭과 건물 윤곽선은 기존 10인치 기준 렌더링의 비율을 실물 크기에 맞춰 적용했다. 도로 선폭은 지도 시각화용이다. 배경은 흰색이고 지도 경계의 획은 출력 면에서 잘린다. 원본 지도 데이터 © OpenStreetMap contributors.

다른 폴더에서 결과를 비교하려면 생성기와 검증기에 같은 `--output-dir`을 전달한다. 입력 좌표 파일은 `--input`, 검증용 기준 이미지는 `--reference`로 지정할 수 있다.

`create_print_map.py`는 EPS·PDF·PNG 세 파일을 갱신한다. ZIP은 별도로 보관하는 배포 시점의 복사본이다. 새로 검증한 파일로 배포본을 갱신할 때는 저장소 루트에서 다음 Python 코드를 실행한다. 기존 ZIP 내부의 `output/` 경로를 유지한다.

```python
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

print_dir = Path("data/print")
with ZipFile(print_dir / "campus_bundle.zip", "w", ZIP_DEFLATED) as bundle:
    for filename in ("KAIST_map_PRINT_2500x2500mm.eps", "KAIST_map_PRINT_preview.pdf",
                     "KAIST_map_PRINT_preview.png"):
        bundle.write(print_dir / "campus" / filename, f"output/{filename}")
```

EPS는 `pt = m × 1000 × 72 / 25.4`로 좌표를 기록한다. 검증기는 EPS의 각 경로 좌표를 m로 역변환하고 색상·선 굵기·출력 순서를 확인한다. PDF 페이지 크기와 벡터 연산자도 확인하고, PNG는 기존 기준 이미지와 동일 해상도에서 RGB 픽셀을 비교한다.

## 검증 결과

| 항목 | 결과 |
| --- | --- |
| EPS 정밀 경계 | `%%HiResBoundingBox: 0 0 7086.6141732283 7086.6141732283`, 2,500.000000 × 2,500.000000mm |
| EPS 정수 경계 | `%%BoundingBox: 0 0 7087 7087`, 바깥쪽 정수 올림 시 2,500.136111 × 2,500.136111mm |
| EPS 용량 | 81,832바이트 |
| EPS 도형 | 건물 60개, 도로 46개와 각 도로의 외곽선, 원형 정문 1개 |
| EPS 좌표 및 축척 | JSON 기반 벡터 경로의 모든 좌표 대조 통과, 모형 1m = 출력 1,000mm |
| EPS 렌더링 | Ghostscript 9.56.0에서 EPS 경계로 한 페이지를 열어 검토. 변환된 페이지 치수는 2,499.9985 × 2,499.9985mm로, 변환기의 소수점 반올림이 반영됨 |
| 검토 PDF | 2,500 × 2,500mm 한 페이지, 벡터 경로 확인 및 Poppler로 렌더링 검토 |
| PNG와 기존 기준 이미지 | 2000 × 2000px에서 RGB 픽셀 100% 일치 |

## 업체 전달 사항

> EPS를 실제 크기 100%로 출력해 주세요. 자동 페이지 맞춤과 확대·축소를 끄고, 분할 출력 후 조립한 완성 지도 면이 2,500 × 2,500mm가 되도록 해 주세요. 이음매 겹침과 재단 여유는 업체의 분할 단계에서 적용해 주세요.

업체 소프트웨어에 2,500.136mm로 표시된다면 정수 `BoundingBox`를 읽은 경우다. 실제 그림 크기는 정밀 경계 `HiResBoundingBox`와 흰 배경 경로로 확인할 수 있다. Illustrator 가져오기와 업체 RIP 및 실물 인쇄는 별도 확인이 필요하다. 색상 인식 실험에는 실제 출력물에서 보이는 색으로 카메라를 점검한다.
