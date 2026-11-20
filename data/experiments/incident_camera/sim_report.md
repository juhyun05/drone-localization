# Simulated grayscale camera check

Camera: 640x480, horizontal FOV 60.0 deg (pinhole guess from settings.py CameraConfig). Frames get blur, motion blur, uneven light, noise and JPEG.
Map: 39 tags of 40 mm, 4 example photos in their slots. Seed 340, 79 s.

## Random views over the whole map (drone tilt up to 10 deg)

| height (m) | frames | tags decoded | wrong ids | pose found | position err mm (median / p90) | height err mm | heading err deg | photos found | loose outline / not a photo |
|---|---|---|---|---|---|---|---|---|---|
| 0.30-0.50 | 45 | 100% (40/40) | 0 | 49% (22/45) | 6.2 / 34.0 | 1.0 / 4.9 | 0.1 / 0.3 | 100% (2/2) | 0 / 0 |
| 0.50-0.70 | 79 | 100% (156/156) | 0 | 70% (55/79) | 10.5 / 41.1 | 2.0 / 15.9 | 0.1 / 0.8 | 80% (12/15) | 1 / 0 |
| 0.70-0.90 | 90 | 99% (327/330) | 0 | 84% (76/90) | 7.8 / 51.3 | 1.2 / 23.8 | 0.0 / 1.1 | 86% (24/28) | 0 / 0 |
| 0.90-1.10 | 82 | 92% (347/376) | 0 | 93% (76/82) | 7.5 / 63.0 | 1.0 / 24.2 | 0.0 / 0.6 | 42% (18/43) | 8 / 0 |
| 1.10-1.31 | 104 | 78% (580/747) | 0 | 93% (97/104) | 5.8 / 85.6 | 1.1 / 17.5 | 0.0 / 0.4 | 12% (7/60) | 9 / 0 |

Tags count as visible when all four corners are in the frame. Wrong ids are ids reported for tags that are not in view at all (misreads).

### Localization with and without a tilt prior (same frames)

| height (m) | tags only: pose found, position err mm (median / p90) | level prior (default): pose found, position err mm (median / p90) | IMU roll/pitch (1 deg noise): pose found, position err mm (median / p90) |
|---|---|---|---|
| 0.30-0.50 | 49% (22/45), 5.0 / 18.9 | 49% (22/45), 6.2 / 34.0 | 49% (22/45), 4.4 / 11.7 |
| 0.50-0.70 | 70% (55/79), 12.2 / 94.5 | 70% (55/79), 10.5 / 41.1 | 70% (55/79), 7.6 / 22.5 |
| 0.70-0.90 | 84% (76/90), 6.5 / 66.3 | 84% (76/90), 7.8 / 51.3 | 84% (76/90), 5.8 / 24.8 |
| 0.90-1.10 | 93% (76/82), 7.1 / 83.6 | 93% (76/82), 7.5 / 63.0 | 93% (76/82), 5.7 / 31.3 |
| 1.10-1.31 | 93% (97/104), 6.1 / 148.0 | 93% (97/104), 5.8 / 85.6 | 93% (97/104), 5.8 / 38.9 |

With one or two small tags in view, tilt is poorly determined and height x tilt error becomes position error. A weak 'camera looks about straight down' prior (default) removes the flipped solutions; the flight controller's roll/pitch (pass gravity_cam to Localizer.solve) does much better.

## Views aimed at the example photos (drone tilt up to 10 deg)

| height (m) | frames | tags decoded | wrong ids | pose found | position err mm (median / p90) | height err mm | heading err deg | photos found | loose outline / not a photo |
|---|---|---|---|---|---|---|---|---|---|
| 0.30-0.50 | 39 | 100% (28/28) | 0 | 62% (24/39) | 7.1 / 16.3 | 4.1 / 8.5 | 0.1 / 0.4 | 97% (31/32) | 0 / 0 |
| 0.50-0.70 | 58 | 100% (106/106) | 0 | 98% (57/58) | 15.3 / 49.2 | 5.0 / 18.2 | 0.1 / 1.1 | 100% (58/58) | 0 / 0 |
| 0.70-0.90 | 63 | 99% (253/255) | 0 | 100% (63/63) | 5.3 / 39.6 | 0.9 / 7.4 | 0.0 / 0.2 | 86% (54/63) | 0 / 0 |

- Picture outline vs. truth (IoU): median 0.95, worst 10 % below 0.79
- Crops turned the right way up (thick bottom bar found): 62% (89/143); flagged upright but wrong: 0 (without the bar, pipeline.py turns the crop by its slot when the pose is known)
- Photo position on the map from the pose: error median / p90 3.0 / 8.0 mm

## Continuous flight (90 s at 10 fps, height 0.6-1.0 m, Lissajous path)

- Frames with a pose from tags: 86% (771/900), position error median / p90 10.5 / 66.0 mm
- Tracker (Kalman): tracking 771, coasting 31, lost 98, not started 0 frames; error median / p90 9.9 / 65.2 mm

## Flights along the roads (0.6 m high, 0.25 m/s, heading along the road)

| tags on the map | frames with a pose from tags | position error mm (median / p90) | frames with a tracked pose | tracked error mm (median / p90) |
|---|---|---|---|---|
| building tags only | 53% (257/482) | 12.8 / 40.1 | 64% (310/482) | 16.8 / 89.5 |
| building + road tags | 85% (408/482) | 13.0 / 39.8 | 92% (442/482) | 13.1 / 48.8 |

Sample frames: [sim_frames/](sim_frames/) (12 grayscale PNGs + ground_truth.json).
