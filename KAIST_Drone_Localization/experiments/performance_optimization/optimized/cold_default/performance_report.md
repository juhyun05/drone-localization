# Live pipeline timing

Input `video_replay` · scenario `stationary` · 60 s per mode · 0 s warmup

| Period | Input FPS | Localization FPS | Processing mean / P95 (ms) | Age mean / P95 (ms) | Global mean / P95 (ms) | Local mean / P95 (ms) | Skipped % | Stale | Match % | Accepted % | Losses | Process CPU % | XY RMSE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unlimited | 25.00 | 13.15 | 75.44 / 86.20 | 21.99 / 42.75 | 729.17 / 729.17 | 50.28 / 57.70 | 47.33 | 0 | 100.00 | 100.00 | 0 | 366.16 | 0.01 |

Age starts at PC JPEG receipt or actual video delivery to the shared slot. Replay delivery lag is measured against the recorded timeline. CPU is process CPU time divided by elapsed time and can exceed 100%. Results completed after the window are marked as drained in CSV. Without warmup the first received frame is included. Headless runs have no GUI render samples.

Match rate uses global/local attempts. Accepted rate uses all processed frames. Losses count transitions into RELOCALIZING. XY RMSE uses accepted measurements with a known fixed reference. The wrong-acceptance rate uses the same reference and flags position error over 5 cm or yaw error over 15 degrees when available. Relocalization time uses the observation timeline for recovered episodes. RSS and CPU resources are recorded in JSON.
