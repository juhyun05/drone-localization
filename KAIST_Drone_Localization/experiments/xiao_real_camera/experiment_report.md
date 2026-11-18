# XIAO 실물 카메라 통합 기록

2026-11-16 실험에서 Arduino CLI가 COM5를 ESP32 Family Device로 열거했고, Espressif 도구가 ESP32-S3 rev v0.2, USB-Serial/JTAG, 8 MB Flash를 확인했다. COM3·COM4는 Bluetooth 가상 포트였다.

| 단계 | 관측 결과 |
| --- | --- |
| USB 및 부트로더 | COM5, USB VID/PID `303A:1001`, Espressif USB-Serial/JTAG |
| 펌웨어 업로드 | Arduino-ESP32 3.3.5로 빌드한 기존 스케치를 업로드하고 각 쓰기 구간의 해시 검증을 통과 |
| 실제 카메라 초기화 | 시리얼 115200 baud에서 8 MB PSRAM과 **OV3660(PID `0x3660`)** 확인 |
| 첫 센서 프레임 | 카메라 드라이버가 320 × 240, JPEG 포맷으로 보고 |
| Wi-Fi 연결 | 외장 안테나 미장착 상태. 첫 시도에서 `WL_DISCONNECTED`. Wi-Fi 이벤트 로그를 추가한 빌드를 올린 뒤 2.4GHz WPA2-Personal 네트워크에서 DHCP로 `192.168.11.27`을 할당받음 |
| HTTP 스트림 | PC에서 80번 포트 TCP 접근 성공. Edge에서는 재생 확인에 이르지 못함. Python은 HTTP 응답 헤더를 기다리다가 타임아웃됐으며, 보드 로그는 `0xB006`(응답 패킷 송신 실패) |
| 수신 벤치마크 | 120초 측정을 시작했으나 약 20.66초 후 재시도 한계로 종료. 디코딩 0프레임. 정상 수신 FPS와 실물 영상 품질은 미측정. [수신 CSV](stream_metrics.csv) |
| 카메라 보정·실물 지도·연속 추적 | 실측 전. 추정 오차, IoU 분포, 추적 복구 시간 미측정 |

Python 회귀 테스트 27개를 통과했다. HSV 진단의 이미지·CSV·설정 저장 및 재사용은 합성 이미지로, 수신 지표 CSV는 로컬 모의 MJPEG로 확인했다.

## 공식 CameraWebServer 예제 재시험

2026-11-16 Arduino-ESP32 3.3.5의 공식 예제에서 XIAO 카메라 모델과 기존 로컬 Wi-Fi 설정을 지정했다. COM5에 예제를 업로드해 플래시 해시 검증과 Wi-Fi 연결을 확인했다. 이 예제가 현재 보드에서 실행 중이다. [CameraWebServer 소스](https://github.com/espressif/arduino-esp32/tree/3.3.5/libraries/ESP32/examples/Camera/CameraWebServer)

| 확인 항목 | 결과 |
| --- | --- |
| 제어 페이지 `/` | 첫 요청 타임아웃, 재시도에서 HTTP 200 |
| 단일 촬영 `/capture` | HTTP 200, 320 × 240 JPEG 디코딩 성공 |
| 연속 영상 `:81/stream` | 30초 동안 70프레임, 평균 2.33 FPS, 연결 중단·JPEG 오류 0회. [수신 CSV](official_stream_metrics.csv) |

## 다음 실험 실행

1. 외장 안테나를 설치할 때는 XIAO 전원을 끈다. 현재 보드에는 실험용 Wi-Fi 설정을 읽은 공식 예제가 설치돼 있다. [펌웨어 안내](../../firmware/xiao_camera_stream/README.md) · [Seeed 안테나 설치 안내](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/#installation-of-antenna)
2. 시리얼 모니터의 `Camera Ready!` 줄에 표시된 주소를 브라우저로 열고 **Start Stream**을 누른다. 브라우저의 스트림을 닫고 Python 수신을 시작한다. 아래 명령은 `KAIST_Drone_Localization`에서 가상 환경을 활성화한 PowerShell 기준이다.
3. 실제 영상의 방향을 확인한다. 같은 높이에서 길이를 아는 수평 구간을 재고 `tools/calibrate_camera.py span`으로 관측 폭을 계산한다. 렌즈 보정은 실제 체커보드 영상을 수집한 뒤 적용한다. [카메라 보정 안내](../../README.md#녹화와-보정)
4. 카메라를 실제 좌표를 아는 5–10개 지점에 수직으로 고정해 아래 진단을 반복한다. 각 지점에 새 출력 디렉터리를 사용한다. 건물이 희박한 지점을 포함하고, 성공률은 전체 시도에서, 위치 RMSE는 채택된 지점에서 계산한다. yaw 오차를 구할 때는 실제 방향을 별도로 측정한다.
5. 정지 추정이 안정되면 같은 높이로 카메라를 천천히 이동하며 연속 추적을 시험한다. `--record`로 영상·타임스탬프·추정 결과를 저장한다.

```powershell
$cameraUrl = 'http://BOARD_IP:81/stream'  # 공식 예제. 자체 펌웨어는 http://BOARD_IP/stream 사용
python tools/test_camera_stream.py --url $cameraUrl --seconds 30 --csv experiments/xiao_real_camera/stream_metrics_antenna.csv
python main.py --camera-test --camera-url $cameraUrl
python main.py --live-localization --camera-url $cameraUrl --global-only
python tools/inspect_camera_frame.py --url $cameraUrl --nominal-span 0.60 --ground-truth 1.73 2.02 --truth-yaw 0 --output experiments/xiao_real_camera/sample_frames/point01
python main.py --live-localization --camera-url $cameraUrl --record experiments/xiao_real_camera/hand_motion.avi
```

명령의 관측 폭, 좌표, yaw는 예시이므로 실측값으로 바꾼다. 진단은 원본, 전처리 영상, 건물 마스크, HSV 분포, 점유율, IoU, 차점과의 차이, 성공 여부와 정합 시간을 저장한다. 정답은 오차 계산에만 사용한다. PC 수신 시간은 실제 촬영 시각과 구분한다.
