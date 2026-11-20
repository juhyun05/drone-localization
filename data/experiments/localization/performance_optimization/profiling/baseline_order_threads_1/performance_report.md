# Live pipeline timing

Input `video_replay` · scenario `stationary` · 8 s per mode

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 ms | 25.11 | 4.25 | 225.85 / 272.17 | 23.47 / 44.33 | — | 89.25 / 118.51 | 82.59 | 0 | 100.00 | 100.00 | 0 | 97.58 | 0.01 |
| Unlimited | 25.10 | 5.87 | 166.56 / 203.82 | 18.93 / 41.36 | — | 63.30 / 77.35 | 76.12 | 0 | 100.00 | 100.00 | 0 | 97.74 | 0.01 |
| Unlimited | 24.98 | 6.12 | 158.76 / 181.70 | 22.16 / 39.41 | — | 58.57 / 70.95 | 75.00 | 0 | 100.00 | 100.00 | 0 | 99.13 | 0.01 |
| 200 ms | 25.08 | 4.74 | 176.29 / 213.61 | 21.99 / 45.55 | — | 68.43 / 88.30 | 80.60 | 0 | 100.00 | 100.00 | 0 | 86.17 | 0.01 |
| 200 ms | 25.11 | 4.50 | 196.19 / 266.74 | 22.76 / 45.89 | — | 75.69 / 102.35 | 81.59 | 0 | 100.00 | 100.00 | 0 | 89.58 | 0.01 |
| Unlimited | 24.98 | 3.50 | 275.33 / 399.13 | 21.09 / 44.50 | — | 114.89 / 179.27 | 85.50 | 0 | 100.00 | 100.00 | 0 | 94.47 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements and the optional fixed reference point.
