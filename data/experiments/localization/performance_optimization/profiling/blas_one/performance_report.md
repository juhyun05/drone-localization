# Live pipeline timing

Input `video_replay` · scenario `stationary` · 10 s per mode

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlimited | 25.09 | 12.09 | 81.71 / 136.27 | 21.05 / 44.97 | — | 51.81 / 87.19 | 51.39 | 0 | 100.00 | 100.00 | 0 | 98.23 | 0.01 |
| Unlimited | 25.09 | 10.90 | 90.62 / 148.27 | 21.20 / 43.84 | — | 57.52 / 94.51 | 56.18 | 0 | 100.00 | 100.00 | 0 | 97.01 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements and the optional fixed reference point.
