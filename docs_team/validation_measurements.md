# Validation of the 10 company measurements

**Status (2026-11-21).** The inventory is from reading the code at tag `validation-freeze-1`. **No measurement has
been validated against hand labels yet**, so every "actual result" below is *not measured*, except process time,
which has the phase labels. The targets are **proposals for the team to approve before any label is scored**; they
must not be changed after a result is seen.

## 1. Inventory

| # | measurement | state | what exists now |
| --- | --- | --- | --- |
| 1 | process time | **implemented** | duration per phase, cycle baseline (median, quartiles) |
| 2 | worker movement path | **partial** | 2D hip path on fixed camera shots only, in pixels without a floor scale, empty on handheld shots. The GVHMR floor map in metres is drawn, and tools/check_outputs.py sums the path per phase, but it is not in metrics.csv or the report |
| 3 | frequent waist bending | **implemented, 2D** | trunk angle in the image, time in 4 bands (under 20, 20 to 45, 45 to 60, over 60 degrees); flagged view dependent |
| 4 | repetitive work | **implemented** | autocorrelation of wrist motion: share of time periodic, period, count; only periods of 0.4 to 2 s |
| 5 | improvable motions | **missing** | no code finds or suggests improvable motions |
| 6 | hand travel distance | **implemented, 2D** | wrist path relative to the body, in pixels and torso lengths; not in metres |
| 7 | tool pickup time | **missing** | no tool or contact detection |
| 8 | object placement time | **missing** | no object or contact detection |
| 9 | two hand cooperation | **partial** | share of time both wrists move, use ratio, speed correlation. This is simultaneous motion, not proof that the hands cooperate |
| 10 | approximate gaze direction | **partial** | head direction in the image from face keypoints (front, left, right, away, down); not eye gaze; **off by default** (not in motion.enabled) |

Summary: 4 implemented (2 of them only in 2D), 3 partial, 3 missing.

## 2. Validation table

| # | measurement | method | ground truth (how a person measures it) | unit | proposed target | actual | known limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | process time | phase boundaries from Engine B zero shot, duration = end minus start | the two annotators' phase boundaries (exists) | s | mean boundary error at most 10 s, each phase duration within 10 % | boundary MAE 8.09 s (flagship); per phase: pending docs_team/validation_segmentation.md | one video; development result |
| 2 | movement path | mid hip on the GVHMR floor at 5 Hz, summed per shot part | walks between floor points whose distance is known | m | within 20 % over a walk | not measured | each shot part has its own world frame; foot and hip jitter adds length; GVHMR scale is estimated |
| 3 | waist bending | trunk angle from shoulders and hips, in 4 bands | a person assigns the band on sampled frames | band | exact band at least 80 %, within one band at least 95 % | not measured | 2D: bending toward the camera is underestimated |
| 4 | repetitive work | autocorrelation over 10 s windows | a person says repetitive or not for 10 s clips | yes/no | F1 at least 0.70 | not measured | misses cycles longer than 2 s |
| 5 | improvable motions | none yet | an industrial engineer rates each suggestion as useful or not | share useful | at least 60 % rated useful | **not implemented** | a judgement, not a measurement |
| 6 | hand travel | wrist path relative to the hip | wrist keypoints in short consecutive frame runs (Label Studio) | torso lengths | wrist within PCK@0.1 for at least 85 %; travel within 20 % | not measured | 2D; occluded hands |
| 7 | tool pickup time | none yet | a person marks each moment a hand closes on a tool | s | events within 1 s: recall and precision at least 0.70 | **not implemented** | needs contact or object detection |
| 8 | object placement time | none yet | a person marks each moment a hand lets go of a placed object | s | events within 1 s: recall and precision at least 0.70 | **not implemented** | needs contact or object detection |
| 9 | two hand cooperation | both wrists above a speed threshold | a person says both, one or no hand works, on 3 s clips | class | agreement at least 75 % | not measured | fast idle motion counts as work; a holding hand counts as idle |
| 10 | gaze direction | yaw from nose, eyes and ears | a person assigns the head direction on sampled frames | class | agreement at least 70 % | not measured | head, not eyes; poor when the face is small or turned away |

## 3. Smallest hand labelled ground truth (short QiSens clips)

Items are drawn with a fixed seed from the frames where the worker is tracked, so no system output decides what is
labelled, and every annotator gets the same items. Two annotators per task give the human agreement too.

| # | task on the **Labels** page | amount | about |
| --- | --- | --- | --- |
| 1 | none needed (phase labels exist) | | |
| 2 | no page: mark 5 straight walks and take their floor distance from a known length in the scene (for example the rail gauge, 1,435 mm, if the track is visible and standard). A short walk over a taped distance with the same camera would be better | 5 walks | 30 min |
| 3 | Waist bending: trunk angle band | 100 frames | 15 min |
| 4 | Repetitive work | 30 clips of 10 s | 15 min |
| 5 | no page: review once suggestions exist | | |
| 6 | Label Studio (existing workflow): wrist keypoints | 5 runs of 10 consecutive frames | 20 min |
| 7 | Tool pickup moments | one 2 minute window | 10 min |
| 8 | Object placement moments | one 2 minute window | 10 min |
| 9 | Two hand cooperation | 40 clips of 3 s | 15 min |
| 10 | Gaze: head direction | 100 frames | 15 min |

**The page.** Open **Labels** in the app (our accounts only), choose the task and the run. Frames and clips are
shown by the app behind the login and are not exported; labels are saved per annotator in
`data/qisens/validation/labels/<run>/<task>/<user>.csv`. The data rules of docs_team/labelling_guide.md section 1 apply:
no screenshots, no downloads, and clear the browser cache after a QiSens session.

**Not built yet:** the scorer that compares these labels with the system. It is small, and should be written after
the team approves the targets above, so that the targets are fixed first.
