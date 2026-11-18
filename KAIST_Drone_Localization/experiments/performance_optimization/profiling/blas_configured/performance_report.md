# Live pipeline timing

Input `video_replay` · scenario `stationary` · 10 s per mode

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlimited | 25.09 | 16.79 | 58.82 / 79.76 | 20.82 / 43.95 | — | 37.07 / 51.86 | 32.67 | 0 | 100.00 | 100.00 | 0 | 97.77 | 0.01 |
| Unlimited | 24.98 | 22.58 | 43.40 / 56.00 | 18.31 / 38.29 | — | 26.53 / 35.25 | 9.20 | 0 | 100.00 | 100.00 | 0 | 98.36 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements and the optional fixed reference point.
