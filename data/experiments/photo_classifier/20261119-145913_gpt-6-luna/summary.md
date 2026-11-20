# Run 20261119-145913_gpt-6-luna

- Model: gpt-6-luna, reasoning effort: low, image detail: auto, language: English
- Photos: photo_gray, scored 18, failed 0

## is_abnormal vs labels.csv

| | pred abnormal | pred normal |
|---|---|---|
| **truth abnormal** | 7 (TP) | 1 (FN) |
| **truth normal** | 0 (FP) | 10 (TN) |

- Accuracy: 17/18 (94%)
- Abnormal recall: 7/8 (88%)
- Abnormal precision: 7/7 (100%)
- False alarms on normal photos: 0/10 (0%)
- Abnormal photos caught by is_abnormal or needs_human_review: 7/8 (88%)

## Cost

- This run: $0.0050 over 18 calls (avg 1857 input / 172 output tokens per call, 43 of them reasoning)
- All logged spend: $0.0752 of $15.00 (calls stop at $12.00)

## Per photo

| id | photo | truth | pred | category | urgency | quality | review | warnings |
|---|---|---|---|---|---|---|---|---|
| 1 | B_normal_4.png | normal | normal | normal | none | good | no | - |
| 2 | C_abnormal_2.png | abnormal | normal ✗ | normal | none | good | no | - |
| 3 | A_abnormal_1.png | abnormal | abnormal | fire | high | good | no | - |
| 4 | C_abnormal_1.png | abnormal | abnormal | possible_illegal_parking | low | good | yes | - |
| 5 | B_abnormal_2.png | abnormal | abnormal | unauthorized_access | medium | good | yes | - |
| 6 | A_abnormal_2.png | abnormal | abnormal | fire | high | good | no | - |
| 7 | D_normal_2.png | normal | normal | normal | none | good | no | - |
| 8 | C_normal_2.jpeg | normal | normal | normal | none | good | no | - |
| 9 | D_abnormal_1.png | abnormal | abnormal | medical_emergency | high | good | yes | - |
| 10 | B_normal_1.png | normal | normal | normal | none | good | no | - |
| 11 | D_abnormal_2.png | abnormal | abnormal | medical_emergency | high | good | yes | - |
| 12 | B_abnormal_1.png | abnormal | abnormal | unauthorized_access | medium | good | yes | - |
| 13 | A_normal_1.jpeg | normal | normal | normal | none | good | no | - |
| 14 | B_normal_3.png | normal | normal | normal | none | good | no | - |
| 15 | B_normal_2.png | normal | normal | normal | none | good | no | - |
| 16 | D_normal_1.png | normal | normal | normal | none | good | no | - |
| 17 | A_normal_2.jpeg | normal | normal | normal | none | good | no | - |
| 18 | C_normal_1.png | normal | normal | normal | none | good | no | - |

## Descriptions

**1** (B_normal_4.png): The scene appears to show routine work or inspection beside building equipment, with a second person stationed at the restricted-area entrance. No intervention is indicated based on the visible scene.

**2** (C_abnormal_2.png): The road appears clear and quiet, with a sedan and an SUV positioned beside opposite curbs. No visible obstruction, unsafe activity, or other incident is apparent, so no action is needed.

**3** (A_abnormal_1.png): Thick smoke is pouring from the upper floors of the E11 building, with fire-response vehicles already on scene. Treat this as an active fire or serious smoke incident and ensure the area is evacuated and kept clear; the source of the smoke is not directly visible.

**4** (C_abnormal_1.png): The foreground car is stopped close to the right curb with a door open, which may create a hazard or indicate an unattended vehicle. Have an officer check the vehicle when convenient; the image does not show why it is stopped or whether anyone is nearby.

**5** (B_abnormal_2.png): Two hooded individuals are at the restricted-area entrance on the right side of the frame, and one appears to be handling equipment beside the wall. Have an officer check the location soon; the image does not establish whether they are authorized workers or what the person is doing.

**6** (A_abnormal_2.png): A substantial amount of smoke is pouring from the building’s upper floors, with emergency vehicles at the entrance. Treat this as an active fire or smoke emergency and dispatch/coordinate an immediate response; the source of the smoke is not visible.

**7** (D_normal_2.png): The scene appears to show ordinary campus activity: a person crossing the road and bicycles parked along the side. No police response is indicated.

**8** (C_normal_2.jpeg): The road appears clear, with a few pedestrians on the left sidewalk and no visible safety or security incident. No action is needed.

**9** (D_abnormal_1.png): A person is lying on the sidewalk near the lower center of the frame and may be injured or unresponsive. Dispatch an officer to check on them and request medical assistance if needed; their condition cannot be determined from the image.

**10** (B_normal_1.png): The restricted-area entrance appears closed and secured, with no visible sign of unauthorized access or damage. Covered items and a ladder are nearby, but nothing in the image clearly indicates an incident; no action is needed.

**11** (D_abnormal_2.png): A person is lying face-down across the crossing in the foreground, apparently unattended. Dispatch an officer to check their condition and secure the area; the image cannot establish whether they are injured or otherwise in distress.

**12** (B_abnormal_1.png): A masked person appears to be manipulating the door lock at the restricted facility entrance, near the center-right of the frame. Have an officer check the entrance and verify whether the person is authorized; the image shows apparent tampering, but their intent and authorization cannot be confirmed.

**13** (A_normal_1.jpeg): The scene shows ordinary activity outside the E11 building, with parked bicycles and a few people and scooters. No visible safety or security issue requires action.

**14** (B_normal_3.png): The people appear to be using the controlled entrance and handling carts outside the building, with no visible sign of danger or unauthorized entry. No action is needed based on this image.

**15** (B_normal_2.png): The service lane and utility equipment appear orderly, with containers stored beside the building near a clearly marked restricted-area entrance. No action is needed based on the visible scene.

**16** (D_normal_1.png): The scene appears to show ordinary pedestrian activity on the sidewalk near the road. No response is needed.

**17** (A_normal_2.jpeg): The building and entrance area appear orderly, with no visible safety or security incident. No action is needed based on this image.

**18** (C_normal_1.png): The campus road and surrounding grounds appear clear and orderly. No response is needed.
