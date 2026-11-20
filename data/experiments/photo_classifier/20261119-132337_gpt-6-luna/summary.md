# Run 20261119-132337_gpt-6-luna

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

- This run: $0.0056 over 18 calls (avg 1857 input / 164 output tokens per call, 31 of them reasoning)
- All logged spend: $0.0678 of $15.00 (calls stop at $12.00)

## Per photo

| id | photo | truth | pred | category | urgency | quality | review | warnings |
|---|---|---|---|---|---|---|---|---|
| 1 | B_normal_4.png | normal | normal | normal | none | good | no | - |
| 2 | C_abnormal_2.png | abnormal | normal ✗ | normal | none | good | no | - |
| 3 | A_abnormal_1.png | abnormal | abnormal | fire | high | good | no | - |
| 4 | C_abnormal_1.png | abnormal | abnormal | suspicious_vehicle | low | good | yes | - |
| 5 | B_abnormal_2.png | abnormal | abnormal | unauthorized_access | high | good | no | - |
| 6 | A_abnormal_2.png | abnormal | abnormal | fire | high | good | yes | - |
| 7 | D_normal_2.png | normal | normal | normal | none | good | no | - |
| 8 | C_normal_2.jpeg | normal | normal | normal | none | good | no | - |
| 9 | D_abnormal_1.png | abnormal | abnormal | medical_emergency | high | good | yes | - |
| 10 | B_normal_1.png | normal | normal | normal | none | good | no | - |
| 11 | D_abnormal_2.png | abnormal | abnormal | medical_emergency | high | good | yes | - |
| 12 | B_abnormal_1.png | abnormal | abnormal | unauthorized_access | high | good | no | - |
| 13 | A_normal_1.jpeg | normal | normal | normal | none | good | no | - |
| 14 | B_normal_3.png | normal | normal | normal | none | good | no | - |
| 15 | B_normal_2.png | normal | normal | normal | none | good | no | - |
| 16 | D_normal_1.png | normal | normal | normal | none | good | no | - |
| 17 | A_normal_2.jpeg | normal | normal | normal | none | good | no | - |
| 18 | C_normal_1.png | normal | normal | normal | none | good | no | - |

## Descriptions

**1** (B_normal_4.png): Two people are visible at the utility area on the right side of the frame: one appears to be working on the piping, while another stands by the restricted-area entrance with a clipboard or tablet. This looks consistent with routine maintenance or inspection, and nothing visibly indicates an immediate safety or security incident.

**2** (C_abnormal_2.png): The scene appears to show two cars parked or stopped at opposite edges of a quiet campus road. No immediate hazard or disruption is visible, so no action is indicated.

**3** (A_abnormal_1.png): Thick smoke is billowing from the mid-to-upper floors of the building, with fire engines and responders visible at the front. Treat this as an active fire and dispatch or coordinate emergency response immediately; the visible smoke and emergency vehicles make the situation clear.

**4** (C_abnormal_1.png): A car is stopped close to the right curb in the foreground with a side door open, which may create a hazard or indicate an unattended vehicle. Have an officer check the vehicle and its surroundings when convenient; the image does not show why it is stopped or whether anyone is nearby.

**5** (B_abnormal_2.png): Two masked individuals appear to be accessing or tampering with utility equipment beside a restricted-area entrance on the right side of the frame. Dispatch an officer to investigate promptly; the visible restricted-area sign and their actions raise concern, although the image alone cannot confirm their intent or authorization.

**6** (A_abnormal_2.png): Dense smoke is pouring from the upper portion of the building, indicating a possible active fire; emergency vehicles are already positioned near the entrance. Dispatch or confirm fire response immediately and keep people clear of the building; the image does not show the source of the smoke or whether flames are present.

**7** (D_normal_2.png): The scene appears to show ordinary campus activity: a person crossing the road and bicycles parked in racks along the right side. No response is needed.

**8** (C_normal_2.jpeg): The scene appears to show ordinary campus activity, with pedestrians on the left sidewalk and an unobstructed road. No action is needed.

**9** (D_abnormal_1.png): A person is lying motionless on the sidewalk in the center-right foreground, with a bag beside them. Dispatch an officer to check on the person and request medical assistance if needed; the image cannot establish whether they are injured or simply resting.

**10** (B_normal_1.png): The frame shows a secured restricted-area building and ordinary service equipment along the side of the structure. No visible entry, tampering, hazard, or other abnormal activity requires action.

**11** (D_abnormal_2.png): A person is prone and appears unresponsive on the crossing in the foreground, creating a possible immediate medical emergency and a roadway hazard. Dispatch an officer to check on them and request medical assistance if needed; the photo cannot establish whether the person is injured or responsive.

**12** (B_abnormal_1.png): A masked individual appears to be attempting to open or tamper with the entrance to the restricted laboratory area, near the center-right of the frame. Dispatch an officer promptly to investigate and secure the access point; the restricted-area sign and apparent lock manipulation make this a significant security concern.

**13** (A_normal_1.jpeg): The scene shows ordinary campus activity around the E11 building, with parked bicycles and a scooter and a few pedestrians. No action is needed.

**14** (B_normal_3.png): The scene appears to show routine controlled entry and handling of covered carts outside the building. No visible sign of forced access, danger, or disruption; no action is needed.

**15** (B_normal_2.png): The service area beside the building appears orderly, with equipment and palletized containers outside the restricted-area door. No visible safety or security incident is apparent, so no action is needed.

**16** (D_normal_1.png): A person is walking normally along the sidewalk near the center of the frame. No immediate safety or security concern is visible, so no action is needed.

**17** (A_normal_2.jpeg): The building entrance and stairs appear orderly, with no visible sign of a safety or security incident. The small dark object near the lower center is indistinct but does not clearly indicate an abnormal situation.

**18** (C_normal_1.png): The campus road runs through the center of the frame and appears clear, with no visible incident or disruption. No action is needed.
