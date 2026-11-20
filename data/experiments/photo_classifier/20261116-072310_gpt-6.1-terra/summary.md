# Run 20261116-072310_gpt-6.1-terra

- Model: gpt-6.1-terra, reasoning effort: low, image detail: auto, language: English
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

- This run: $0.0532 over 18 calls (avg 1857 input / 129 output tokens per call, 13 of them reasoning)
- All logged spend: $0.0590 of $15.00 (calls stop at $12.00)

## Per photo

| id | photo | truth | pred | category | urgency | quality | review | warnings |
|---|---|---|---|---|---|---|---|---|
| 1 | B_normal_4.png | normal | normal | normal | none | good | no | - |
| 2 | C_abnormal_2.png | abnormal | abnormal | possible_illegal_parking | low | good | yes | - |
| 3 | A_abnormal_1.png | abnormal | abnormal | fire | high | good | no | - |
| 4 | C_abnormal_1.png | abnormal | abnormal | roadway_obstruction | low | good | yes | - |
| 5 | B_abnormal_2.png | abnormal | abnormal | suspected_unauthorized_access | medium | good | yes | - |
| 6 | A_abnormal_2.png | abnormal | abnormal | fire | high | good | no | - |
| 7 | D_normal_2.png | normal | normal | normal | none | good | no | - |
| 8 | C_normal_2.jpeg | normal | normal | normal | none | good | no | - |
| 9 | D_abnormal_1.png | abnormal | abnormal | possible_medical_emergency | high | good | yes | - |
| 10 | B_normal_1.png | normal | normal | normal | none | good | no | - |
| 11 | D_abnormal_2.png | abnormal | abnormal | possible_medical_emergency | high | good | yes | - |
| 12 | B_abnormal_1.png | abnormal | abnormal | possible_unauthorized_access | medium | good | yes | - |
| 13 | A_normal_1.jpeg | normal | normal | normal | none | good | no | - |
| 14 | B_normal_3.png | normal | normal | normal | none | good | no | - |
| 15 | B_normal_2.png | normal | normal | normal | none | good | no | - |
| 16 | D_normal_1.png | normal | normal | normal | none | good | no | - |
| 17 | A_normal_2.jpeg | normal | normal | normal | none | good | no | - |
| 18 | C_normal_1.png | normal | normal | normal | none | good | no | - |

## Descriptions

**1** (B_normal_4.png): Two people appear to be working near equipment and a restricted-area entrance on the right side of the frame. No immediate police action is indicated; their authorization cannot be determined from the photo.

**2** (C_abnormal_2.png): Two vehicles appear parked along the road edges in the foreground, possibly in a restricted area. Check the local parking restrictions and vehicles when convenient; the sign text and whether the vehicles are occupied are not clear.

**3** (A_abnormal_1.png): Heavy smoke from the upper floors of E11 indicates a probable building fire; firefighters are already at the scene. Dispatch campus police immediately to support evacuation, secure the area, and coordinate with fire crews.

**4** (C_abnormal_1.png): A car is stopped partly in the roadway with a door open toward the sidewalk. An officer should review and check it when convenient; the image does not establish how long it has been stopped or why the door is open.

**5** (B_abnormal_2.png): Two face-covered people are near a restricted-area entrance on the right side of the frame, with one reaching toward building pipes. Send an officer to check their authorization and the equipment; their identities and intentions are not clear from the image.

**6** (A_abnormal_2.png): Heavy smoke is coming from the upper floors of the building, indicating a likely fire; no flames are visible. Dispatch police immediately, coordinate with the emergency crews visible at the entrance, and keep people away from the building.

**7** (D_normal_2.png): The scene shows ordinary pedestrian and parking activity on a campus road. No police action is indicated.

**8** (C_normal_2.jpeg): Pedestrians are walking along the left sidewalk beside a mostly empty road. No visible incident requires police action.

**9** (D_abnormal_1.png): A person is lying on the sidewalk in the lower center of the frame and may need urgent assistance. Dispatch an officer to check their responsiveness and request medical assistance as needed; their condition cannot be determined from the photo.

**10** (B_normal_1.png): The restricted-area entrance on the right appears closed, with no visible unauthorized activity. No police action is indicated from the photo.

**11** (D_abnormal_2.png): A person is lying on the road crossing in the lower center of the image and may be injured or unresponsive. Dispatch an officer and request medical assistance; check responsiveness and protect the person from traffic. Their condition and reason for lying down cannot be determined from the photo.

**12** (B_abnormal_1.png): A person appears to be manipulating the access point at the restricted-area door on the right side of the frame. An officer should check the location and verify authorization; the photo alone does not establish whether the person has permission to enter.

**13** (A_normal_1.jpeg): The area around the E11 building appears to show ordinary campus activity. No police action is recommended based on the visible scene.

**14** (B_normal_3.png): People appear to be working and using an access-controlled entrance on the right side of the frame. No clear safety or security incident is visible, so no police action is recommended.

**15** (B_normal_2.png): The building exterior and stored containers appear undisturbed in the center-right of the frame. No police action is indicated by the visible scene.

**16** (D_normal_1.png): A pedestrian is walking along the sidewalk near the center of the frame. No apparent safety or security issue is visible, and no action is recommended.

**17** (A_normal_2.jpeg): The photo shows a building exterior and entrance area, with exposed ground and materials in the foreground. No immediate safety or security incident is visible; no action is recommended.

**18** (C_normal_1.png): The road and surrounding campus grounds appear orderly, with no visible incident. No police action is recommended.
