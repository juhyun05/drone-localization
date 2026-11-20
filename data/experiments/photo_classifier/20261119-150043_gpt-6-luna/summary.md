# Run 20261119-150043_gpt-6-luna

- Model: gpt-6-luna, reasoning effort: low, image detail: auto, language: English
- Photos: /private/tmp/claude-501/-Users-chaejeonghyeon/a336d310-739a-4b74-a229-a985412db9c9/scratchpad/photo_cam120, scored 18, failed 0

## is_abnormal vs labels.csv

| | pred abnormal | pred normal |
|---|---|---|
| **truth abnormal** | 6 (TP) | 2 (FN) |
| **truth normal** | 0 (FP) | 10 (TN) |

- Accuracy: 16/18 (89%)
- Abnormal recall: 6/8 (75%)
- Abnormal precision: 6/6 (100%)
- False alarms on normal photos: 0/10 (0%)
- Abnormal photos caught by is_abnormal or needs_human_review: 6/8 (75%)

## Cost

- This run: $0.0036 over 18 calls (avg 699 input / 255 output tokens per call, 126 of them reasoning)
- All logged spend: $0.0788 of $15.00 (calls stop at $12.00)

## Per photo

| id | photo | truth | pred | category | urgency | quality | review | warnings |
|---|---|---|---|---|---|---|---|---|
| 1 | B_normal_4.png | normal | normal | normal | none | degraded | no | - |
| 2 | C_abnormal_2.png | abnormal | normal ✗ | normal | none | degraded | no | - |
| 3 | A_abnormal_1.png | abnormal | abnormal | possible_fire | medium | degraded | yes | - |
| 4 | C_abnormal_1.png | abnormal | normal ✗ | normal | none | degraded | no | - |
| 5 | B_abnormal_2.png | abnormal | abnormal | possible_person_in_distress | medium | degraded | yes | - |
| 6 | A_abnormal_2.png | abnormal | abnormal | possible_fire | high | degraded | yes | - |
| 7 | D_normal_2.png | normal | normal | normal | none | degraded | no | - |
| 8 | C_normal_2.jpeg | normal | normal | normal | none | degraded | no | - |
| 9 | D_abnormal_1.png | abnormal | abnormal | possible_person_down | medium | degraded | yes | - |
| 10 | B_normal_1.png | normal | normal | normal | none | degraded | no | - |
| 11 | D_abnormal_2.png | abnormal | abnormal | possible_person_down | medium | degraded | yes | - |
| 12 | B_abnormal_1.png | abnormal | abnormal | suspicious_activity | medium | degraded | yes | - |
| 13 | A_normal_1.jpeg | normal | normal | normal | none | degraded | no | - |
| 14 | B_normal_3.png | normal | normal | normal | none | degraded | no | - |
| 15 | B_normal_2.png | normal | normal | normal | none | degraded | no | - |
| 16 | D_normal_1.png | normal | normal | normal | none | degraded | no | - |
| 17 | A_normal_2.jpeg | normal | normal | normal | none | degraded | no | - |
| 18 | C_normal_1.png | normal | normal | normal | none | degraded | no | - |

## Descriptions

**1** (B_normal_4.png): The scene appears to show ordinary activity beside the building, with people standing near the entrances. No clear safety or security incident is visible; distance and low image detail make it difficult to identify the people or nearby items precisely.

**2** (C_abnormal_2.png): The scene appears to show an ordinary roadway with cars near the sides and no visible incident requiring action. The image is blurry and distant, but no specific abnormal situation is apparent.

**3** (A_abnormal_1.png): A plume-like haze is visible near the upper-right side of the building and could be smoke, but the low-resolution image does not show its source or any flames clearly. Have an officer verify the area promptly and check for signs of fire.

**4** (C_abnormal_1.png): A vehicle is on the curved road near the lower center of the frame, with no visible sign of an incident or disruption. No action is indicated; the distant, low-resolution view limits detail.

**5** (B_abnormal_2.png): Near the building on the right side of the frame, a person appears bent or crouched beside a wheeled object, with another indistinct figure or object nearby. Have an officer check the area soon to confirm whether anyone needs assistance; the low-resolution image makes it unclear whether this is an emergency or ordinary activity.

**6** (A_abnormal_2.png): A plume that may be smoke is visible near the upper-right side of the structure. Dispatch an officer to check for fire or another hazard; the image is too unclear to confirm the source.

**7** (D_normal_2.png): The scene appears to show an ordinary pedestrian on a tree-lined path beyond the foreground crosswalk. No response is indicated; the distant figure and overall scene are somewhat blurry, but nothing visibly suggests an incident.

**8** (C_normal_2.jpeg): The road appears clear in the visible frame, with no apparent incident or obstruction. No action is needed; the distant, low-resolution view limits detail, but nothing visible suggests an abnormal situation.

**9** (D_abnormal_1.png): Several dark shapes lie on the path near the center of the frame; one could possibly be a person on the ground, but the low resolution makes this uncertain. Have an officer check the area soon to confirm whether anyone needs assistance.

**10** (B_normal_1.png): The scene appears to show an ordinary building exterior and walkway, with parked vehicles at the left. No police action is indicated; the low resolution limits detail, but nothing visibly suggests an incident.

**11** (D_abnormal_2.png): A dark shape lies near the center of the crossing in the foreground and could be a person or an object. Have an officer check the location soon; the image is too blurred and distant to determine what it is.

**12** (B_abnormal_1.png): One person appears to be reaching toward a building entrance or fixture, while another is nearby. The image is too unclear to tell whether this is authorized activity or an attempt to access the building; have an officer review the scene and check the entrance.

**13** (A_normal_1.jpeg): The image shows a campus building and surrounding grounds, with no visible incident requiring action. The low resolution limits fine detail, but nothing apparent suggests an abnormal situation.

**14** (B_normal_3.png): People are gathered near the building entrance, with a vehicle or equipment at the left edge. No clear safety or security issue is apparent, though the distant, low-resolution view limits detail.

**15** (B_normal_2.png): The scene appears to show an ordinary building-side access lane with equipment and a vehicle in the distance. No response is indicated; fine details are difficult to make out because the image is low resolution.

**16** (D_normal_1.png): A person is walking along the path near the center of the frame, with no visible sign of an emergency or suspicious activity. No action is indicated; the image is somewhat distant and low-resolution, but the visible scene appears ordinary.

**17** (A_normal_2.jpeg): The image shows the exterior of a large building, with no clearly visible safety or security incident. The distant figures or shapes at the left are too indistinct to identify, but nothing visible suggests an immediate concern.

**18** (C_normal_1.png): The road or path appears clear in the visible frame, with no apparent incident requiring action. The image is somewhat low-resolution, but no specific hazard is evident.
