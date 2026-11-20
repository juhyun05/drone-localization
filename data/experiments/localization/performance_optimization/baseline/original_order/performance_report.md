# Live pipeline timing

Input `video_replay` · scenario `stationary` · 60 s per mode

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlimited | 25.01 | 2.58 | 385.91 / 759.86 | 23.71 / 44.50 | 2825.99 / 2825.99 | 201.76 / 429.61 | 89.61 | 0 | 100.00 | 100.00 | 0 | 253.33 | 0.01 |
| 100 ms | 25.00 | 3.65 | 272.05 / 416.35 | 22.87 / 45.29 | 2476.68 / 2476.68 | 136.11 / 242.77 | 85.33 | 0 | 100.00 | 100.00 | 0 | 306.79 | 0.01 |
| 150 ms | 25.00 | 4.20 | 236.86 / 381.61 | 23.35 / 44.58 | 2431.50 / 2431.50 | 112.71 / 198.91 | 83.13 | 0 | 100.00 | 100.00 | 0 | 315.53 | 0.01 |
| 200 ms | 25.00 | 4.17 | 222.00 / 312.05 | 21.25 / 44.82 | 2395.87 / 2395.87 | 106.79 / 161.93 | 83.33 | 0 | 100.00 | 100.00 | 0 | 314.15 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements and the optional fixed reference point.
