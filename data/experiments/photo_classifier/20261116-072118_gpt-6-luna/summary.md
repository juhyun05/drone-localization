# Run 20261116-072118_gpt-6-luna

- Model: gpt-6-luna, reasoning effort: low, image detail: auto, language: English
- Photos: photo_unlabeled, scored 18, failed 0

## is_abnormal vs labels.csv

| | pred abnormal | pred normal |
|---|---|---|
| **truth abnormal** | 8 (TP) | 0 (FN) |
| **truth normal** | 0 (FP) | 10 (TN) |

- Accuracy: 18/18 (100%)
- Abnormal recall: 8/8 (100%)
- Abnormal precision: 8/8 (100%)
- False alarms on normal photos: 0/10 (0%)
- Abnormal photos caught by is_abnormal or needs_human_review: 8/8 (100%)

## Cost

- This run: $0.0055 over 18 calls (avg 1857 input / 178 output tokens per call, 48 of them reasoning)
- All logged spend: $0.0059 of $15.00 (calls stop at $12.00)

## Per photo

| id | photo | truth | pred | category | urgency | quality | review | warnings |
|---|---|---|---|---|---|---|---|---|
| 1 | B_normal_4.png | normal | normal | normal | none | good | no | - |
| 2 | C_abnormal_2.png | abnormal | abnormal | illegal_parking | low | good | yes | - |
| 3 | A_abnormal_1.png | abnormal | abnormal | fire | high | good | no | - |
| 4 | C_abnormal_1.png | abnormal | abnormal | vehicle_obstruction | low | good | yes | - |
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

**1** (B_normal_4.png): The scene appears to show routine utility work beside a restricted-area building, with a uniformed person nearby. No response is needed based on the visible scene.

**2** (C_abnormal_2.png): Two vehicles appear to be parked or stopped at the road edges, partially occupying the roadway. Check whether they are authorized or temporarily stopped; the image does not establish how long they have been there or whether parking is prohibited at this location.

**3** (A_abnormal_1.png): Smoke is coming from the upper-middle floors of the E11 building, with fire crews and people gathered outside. Dispatch and coordinate with responding firefighters; the visible smoke and emergency response indicate an active fire or suspected fire.

**4** (C_abnormal_1.png): The silver sedan in the lower-right portion of the frame is stopped at the road edge with a door open. Check whether it is safely loading or unloading, since no person is visible and the reason for the stop is unclear.

**5** (B_abnormal_2.png): Two masked individuals are at the restricted-area entrance on the right side of the frame, with one reaching toward the building’s exterior pipes or equipment. Have an officer check the area soon; their intent and whether any equipment has been tampered with are not clear from the photo.

**6** (A_abnormal_2.png): Thick smoke is pouring from the upper floors of the building, and emergency vehicles are already at the entrance. Dispatch or coordinate an immediate fire response and keep people clear; the source of the smoke is obscured.

**7** (D_normal_2.png): The scene shows a person crossing the road at the marked crossing, with parked bicycles and vehicles in the background. No abnormal activity or immediate hazard is visible, so no action is needed.

**8** (C_normal_2.jpeg): The scene appears to show ordinary campus activity, with pedestrians using the left sidewalk and the road clear. No action is needed.

**9** (D_abnormal_1.png): A person is lying on the sidewalk near the lower center of the frame and may be unwell or injured. Dispatch an officer to check their condition promptly; the image does not show whether they are conscious or what caused them to be on the ground.

**10** (B_normal_1.png): The restricted facility entrance appears secured, with access-control signage, a camera, and a fenced perimeter. Covered items are staged along the path, but nothing visible indicates unauthorized entry or an immediate hazard; no action is needed based on this image.

**11** (D_abnormal_2.png): A person is down on the pedestrian crossing in the foreground, with a bag beside them. Dispatch an officer to check their condition and protect them from traffic; the photo cannot show whether they are injured or responsive.

**12** (B_abnormal_1.png): A masked person appears to be manipulating the entrance to a clearly marked restricted facility, with a backpack beside them. Have an officer check the scene soon and verify whether the person is authorized; the image shows the activity but cannot establish intent or authorization.

**13** (A_normal_1.jpeg): The scene shows ordinary activity outside the campus building, with pedestrians and parked bicycles and a motorcycle. No action is needed.

**14** (B_normal_3.png): People appear to be carrying out routine work and entering the restricted-area doorway using an access card. No apparent safety or security issue is visible, so no action is needed.

**15** (B_normal_2.png): The service area beside the utility building appears orderly, with equipment and containers stored near the restricted-area door. No visible safety or security incident warrants action.

**16** (D_normal_1.png): A person is walking normally on the sidewalk near the center of the frame, beside the road. No response is needed; the visible scene appears ordinary.

**17** (A_normal_2.jpeg): The building and exterior steps appear undisturbed, with no visible safety or security issue requiring action. No response is needed.

**18** (C_normal_1.png): The campus road and surrounding grounds appear orderly, with no visible incident or obstruction. No action is needed.
