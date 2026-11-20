# Live pipeline timing

Input `video_replay` · scenario `stationary` · 8 s per mode

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 ms | 25.12 | 4.50 | 195.05 / 288.15 | 21.05 / 47.22 | — | 95.51 / 146.09 | 81.59 | 0 | 100.00 | 100.00 | 0 | 258.00 | 0.01 |
| Unlimited | 24.98 | 5.50 | 180.00 / 255.54 | 22.44 / 44.50 | — | 85.60 / 128.71 | 77.50 | 0 | 100.00 | 100.00 | 0 | 295.52 | 0.01 |
| Unlimited | 25.09 | 5.37 | 181.09 / 233.16 | 22.22 / 45.59 | — | 85.29 / 115.74 | 78.11 | 0 | 100.00 | 100.00 | 0 | 272.04 | 0.01 |
| 200 ms | 24.97 | 4.74 | 173.37 / 238.97 | 19.29 / 31.85 | — | 83.02 / 119.08 | 80.50 | 0 | 100.00 | 100.00 | 0 | 253.58 | 0.01 |
| 200 ms | 25.08 | 4.49 | 209.84 / 254.80 | 21.04 / 43.19 | — | 104.68 / 136.64 | 81.59 | 0 | 100.00 | 100.00 | 0 | 274.68 | 0.01 |
| Unlimited | 24.96 | 4.37 | 220.01 / 294.41 | 22.89 / 45.10 | — | 110.14 / 175.15 | 82.00 | 0 | 100.00 | 100.00 | 0 | 278.99 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements and the optional fixed reference point.
