# Live pipeline timing

Input `video_replay` · scenario `stationary` · 60 s per mode · 4 s warmup

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. 150 ms | 25.00 | 6.42 | 71.24 / 84.63 | 19.04 / 46.41 | — | 43.70 / 54.03 | 74.33 | 0 | 100.00 | 100.00 | 0 | 47.26 | 0.01 |
| 1. 100 ms | 25.01 | 9.16 | 68.91 / 82.80 | 22.48 / 46.83 | — | 42.82 / 53.43 | 63.29 | 0 | 100.00 | 100.00 | 0 | 64.31 | 0.01 |
| 1. 200 ms | 25.00 | 4.83 | 109.22 / 235.21 | 17.99 / 38.35 | — | 72.26 / 145.87 | 80.60 | 0 | 100.00 | 100.00 | 0 | 53.85 | 0.01 |
| 1. Unlimited | 25.00 | 16.33 | 60.49 / 108.36 | 22.39 / 43.60 | — | 38.61 / 73.03 | 34.60 | 0 | 100.00 | 100.00 | 0 | 99.39 | 0.01 |
| 2. 100 ms | 25.00 | 9.13 | 48.47 / 83.79 | 23.14 / 46.84 | — | 30.36 / 54.48 | 63.40 | 0 | 100.00 | 100.00 | 0 | 47.44 | 0.01 |
| 2. 200 ms | 24.98 | 4.92 | 38.93 / 50.81 | 22.15 / 46.63 | — | 24.70 / 33.62 | 80.32 | 0 | 100.00 | 100.00 | 0 | 20.34 | 0.01 |
| 2. Unlimited | 25.00 | 17.96 | 55.02 / 80.52 | 21.52 / 43.36 | — | 34.75 / 52.83 | 28.07 | 0 | 100.00 | 100.00 | 0 | 101.29 | 0.01 |
| 2. 150 ms | 25.01 | 6.42 | 43.34 / 81.70 | 21.47 / 46.68 | — | 27.43 / 52.64 | 74.28 | 0 | 100.00 | 100.00 | 0 | 30.21 | 0.01 |
| 3. 200 ms | 24.99 | 4.92 | 41.05 / 58.19 | 17.39 / 32.12 | — | 25.82 / 37.83 | 80.27 | 0 | 100.00 | 100.00 | 0 | 22.08 | 0.01 |
| 3. Unlimited | 25.01 | 18.25 | 54.19 / 79.26 | 21.73 / 43.35 | — | 34.14 / 52.08 | 26.98 | 0 | 100.00 | 100.00 | 0 | 99.92 | 0.01 |
| 3. 150 ms | 25.00 | 6.18 | 77.04 / 199.97 | 23.08 / 46.97 | — | 48.88 / 131.15 | 75.27 | 0 | 100.00 | 100.00 | 0 | 47.44 | 0.01 |
| 3. 100 ms | 25.00 | 9.15 | 55.97 / 95.57 | 23.02 / 46.94 | — | 33.66 / 64.56 | 63.40 | 0 | 100.00 | 100.00 | 0 | 53.17 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Without warmup the first received frame is included. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements with a known fixed reference. The wrong-acceptance rate uses the same reference and flags position error over 5 cm or yaw error over 15 degrees when available. Relocalization time uses the observation timeline for recovered episodes. RSS and CPU resources are recorded in JSON.
