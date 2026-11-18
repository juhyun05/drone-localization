# Live pipeline timing

Input `video_replay` · scenario `stationary` · 60 s per mode

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlimited | 25.01 | 10.68 | 93.13 / 108.02 | 22.22 / 43.55 | 684.19 / 684.19 | 47.18 / 56.49 | 57.23 | 0 | 100.00 | 100.00 | 0 | 377.76 | 0.01 |
| 100 ms | 25.00 | 8.80 | 106.13 / 125.95 | 22.04 / 46.09 | 1015.75 / 1015.75 | 54.44 / 65.35 | 64.73 | 0 | 100.00 | 100.00 | 0 | 355.93 | 0.01 |
| 150 ms | 25.00 | 6.28 | 104.45 / 127.77 | 20.24 / 46.99 | 1062.02 / 1062.02 | 54.60 / 68.55 | 74.80 | 0 | 100.00 | 100.00 | 0 | 242.50 | 0.01 |
| 200 ms | 25.01 | 4.30 | 206.12 / 293.92 | 19.61 / 42.46 | 1026.35 / 1026.35 | 107.76 / 159.71 | 82.81 | 0 | 100.00 | 100.00 | 0 | 317.51 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements and the optional fixed reference point.

## 실험 조건

320 × 240 해상도의 정지 합성 영상이다. 기본 SyntheticCamera를 seed 42로 초기화하고 지도 좌표 (1.73, 2.02) m, yaw 37°의 영상 한 장을 생성했다. 같은 픽셀을 1,625프레임의 MJPG AVI로 기록하고, 사이드카에 25 FPS에 해당하는 `i / 25` 초의 시간표를 기록했다. 네 모드에서 같은 파일을 처음부터 재생했고 자동 축척, combined 조명 보정 및 그림자 가중 정합을 켰다. 각 모드의 전역 정합 표본은 초기 탐색 1회다.

동일 영상 조건에서 Unlimited가 위치 추정 처리율이 가장 높았고 P95 Frame Age는 43.55ms였다. 150ms는 처리율이 6.28 FPS로 낮아졌고 프로세스 CPU가 242.50%로 측정됐다. 네 모드의 위치 RMSE는 0.71cm, 평균 yaw 오차는 0.5°였다. 이 정지 합성 재생으로 확인한 오차는 실물 카메라와 이동 장면의 오차를 대표하지 않는다.

## 회귀 검증

전체 자동 테스트 58개를 통과했다. 기존 합성 추적 벤치마크는 변경 전후 모두 137개 입력 중 125개 관측을 채택했고, 필터 위치 오차가 5cm 이내인 프레임은 131개였다. 새 스케줄러에 20 FPS의 약 7초 이동 합성 영상을 재생한 별도 확인에서는 150ms 설정으로 140개 중 43개를 처리·채택했고, 채택 관측의 위치 RMSE는 0.53cm였다.

## XIAO 카메라

기존 실험의 스트림 주소에 약 15초 동안 연결을 시도했으나 타임아웃됐고, COM5에서 현재 IP가 출력되는 것을 확인하지 못했다. 실물 4개 모드의 성능 및 최적 주기 결정은 현재 스트림 주소에서 프레임이 수신된 뒤 진행할 수 있다. 첫 실물 측정의 기준 모드는 기본값인 Unlimited다.
