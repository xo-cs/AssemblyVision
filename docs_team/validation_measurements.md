# Validation of the 10 company measurements

**Status (2026-11-26).** All 10 have an implementation. **No measurement has been validated against hand labels yet**,
so every "actual result" below is *not measured*, except process time, which has the phase labels. The targets are
**proposals under review by Khurshid**; they are fixed before any label is scored, and the scorer is not written until
then. Everything marked **experimental** stays so until it has ground truth.

The added measures (cycle times, floor path, suggestions, tool and object events) are computed from a finished run by
output/extras.py. They do not change steps.csv or metrics.csv of a run; the steps stay byte identical to tag
`validation-freeze-1` (checked on 6 public runs). They appear in Results and in the PDF, and the report's metrics table
carries them with their flags. The report is outside the frozen method (Khurshid's decision of 2026-11-26).

## 1. Inventory

| # | measurement | state | what exists now |
| --- | --- | --- | --- |
| 1 | process time | **implemented** | duration per phase, cycle baseline, cycle time graph (one bar per cycle when a phase repeats) |
| 2 | worker movement path | **implemented, not validated** | `movement_path_floor`: mid hip on the GVHMR floor in metres at 5 Hz, per phase, inside each shot part; needs body3d. The older 2D path stays for fixed shots |
| 3 | frequent waist bending | **implemented, 2D** | trunk angle in the image, time in 4 bands (under 20, 20 to 45, 45 to 60, over 60 degrees); flagged view dependent |
| 4 | repetitive work | **implemented** | autocorrelation of wrist motion: share of time periodic, period, count; only periods of 0.4 to 2 s |
| 5 | improvable motions | **experimental** | `improvable_motions`: a rule list on the metrics above; each hit is a suggestion with metric, value, threshold and time ranges |
| 6 | hand travel distance | **implemented, 2D** | wrist path relative to the body, in pixels and torso lengths; not in metres |
| 7 | tool pickup time | **experimental** | `tool_pickup_count`, `tool_pickup_time_s` (mean reach time): a reach ends at a detected object, then the object leaves with the hand |
| 8 | object placement time | **experimental** | `object_place_count`, `object_place_time_s`: a reach ends at a detected object, then the hand leaves and the object stays |
| 9 | two hand cooperation | **partial** | share of time both wrists move, use ratio, speed correlation. This is simultaneous motion, not proof that the hands cooperate |
| 10 | approximate gaze direction | **partial** | head direction in the image from face keypoints (front, left, right, away, down); not eye gaze; **off by default** (not in motion.enabled) |

**Measures 7 and 8 found nothing on the public runs so far:** 0 pickups and 0 placements on 3 runs (about 7 minutes of
video, including a harness bench). The detector (YOLO26 m, COCO, CPU) has almost no workshop classes: no wrench, hammer
or drill. It found 2 portable objects in 149 frames of the bench clip. **So a count of 0 does not mean that no tool was
picked up.** These two measures are unlikely to be useful without a detector trained on the actual tools.

**Rules behind the suggestions** (team defaults that flag what a person should look at; not ergonomic limits from a
standard; not validated): deep bending (over 60 degrees for at least 10 % of the measured time), long bending (over 45
degrees for at least 25 %), long idle (idle share at least 0.25), low two hand use (below 0.25), repetitive motion (share
at least 0.50), and long phase, long hand travel and long walking (at least 1.5 times the median of the run's phases, with
at least 3 phases; walking also at least 5 m). A metric measured on less than half of the phase gives no suggestion.

## 2. Validation table

| # | measurement | method | ground truth (how a person measures it) | unit | proposed target | actual | known limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | process time | phase boundaries from Engine B zero shot, duration = end minus start | the two annotators' phase boundaries (exists, two annotators) | s | mean boundary error at most 10 s, each phase duration within 10 % | boundary MAE 8.09 s (flagship); see docs_team/validation_segmentation.md | one video; development result |
| 2 | movement path | mid hip on the GVHMR floor at 5 Hz, summed per shot part | walks between floor points whose distance is known | m | within 20 % over a walk | not measured | each shot part has its own world frame, so walks across parts are not joined; jitter adds length; GVHMR scale is estimated |
| 3 | waist bending | trunk angle from shoulders and hips, in 4 bands | a person assigns the band on sampled frames (one annotator, Khurshid) | band | exact band at least 80 %, within one band at least 95 % | not measured | 2D: bending toward the camera is underestimated |
| 4 | repetitive work | autocorrelation over 10 s windows | a person says repetitive or not for 10 s clips | yes/no | F1 at least 0.70 | not measured | misses cycles longer than 2 s |
| 5 | improvable motions (**experimental**) | rule list on the measured metrics | an industrial engineer rates each suggestion as useful or not | share useful | at least 60 % rated useful | not measured | a judgement; the thresholds are defaults; inherits every limit of the metrics it uses |
| 6 | hand travel | wrist path relative to the hip | wrist keypoints in short consecutive frame runs (Label Studio) | torso lengths | wrist within PCK@0.1 for at least 85 %; travel within 20 % | not measured | 2D; occluded hands |
| 7 | tool pickup time (**experimental**) | wrist speed drops at a detected object (YOLO26 m, COCO, 2 per second), and the object then moves with the hand; time = the reach | a person marks each moment a hand closes on a tool | s | events within 1 s: recall and precision at least 0.70 | not measured | COCO has almost no workshop tools, so most pickups are missed; 2D distance; occlusion |
| 8 | object placement time (**experimental**) | wrist speed drops at a detected object, and the hand then leaves while the object stays; time = the reach | a person marks each moment a hand lets go of a placed object | s | events within 1 s: recall and precision at least 0.70 | not measured | as for 7; a hand merely passing a static object can count |
| 9 | two hand cooperation | both wrists above a speed threshold | a person says both, one or no hand works, on 3 s clips | class | agreement at least 75 % | not measured | fast idle motion counts as work; a holding hand counts as idle |
| 10 | gaze direction | yaw from nose, eyes and ears | a person assigns the head direction on sampled frames | class | agreement at least 70 % | not measured | head, not eyes; poor when the face is small or turned away |

## 3. Smallest hand labelled ground truth (short QiSens clips)

Items are drawn with a fixed seed from the frames where the worker is tracked, so no system output decides what is
labelled, and every annotator gets the same items. Two annotators per task would also give the human agreement.

| # | task on the **Labels** page | amount | about |
| --- | --- | --- | --- |
| 1 | none needed (phase labels exist) | | |
| 2 | no page: mark 5 straight walks and take their floor distance from a known length in the scene (for example the rail gauge, 1,435 mm, if the track is visible and standard). A short walk over a taped distance with the same camera would be better | 5 walks | 30 min |
| 3 | Waist bending: trunk angle band | 100 frames | 15 min |
| 4 | Repetitive work | 30 clips of 10 s | 15 min |
| 5 | no page: an engineer rates the suggestions in the PDF | all suggestions of the flagship | 20 min |
| 6 | Label Studio (existing workflow): wrist keypoints | 5 runs of 10 consecutive frames | 20 min |
| 7 | Tool pickup moments | one 2 minute window | 10 min |
| 8 | Object placement moments | one 2 minute window | 10 min |
| 9 | Two hand cooperation | 40 clips of 3 s | 15 min |
| 10 | Gaze: head direction | 100 frames | 15 min |

**The page.** The Labels page is no longer in the sidebar (2026-11-26); its code and all saved labels are kept. Labels
are saved per annotator in `data/qisens/validation/labels/<run>/<task>/<user>.csv`. The data rules of
docs_team/labelling_guide.md section 1 apply: no screenshots, no downloads, and clear the browser cache after a QiSens session.

**Labels made so far.** Khurshid labelled the measurement tasks on the Labels page in the week of 2026-11-19, alone:
**one annotator**, so no human agreement can be reported for these tasks and the human ceiling is unknown. The files
are under `data/qisens/validation/labels/` (Claude cannot list that folder). To see which tasks have how many labels,
without showing any value:

```
cd ~/AssemblyVision && find data/qisens/validation/labels -name '*.csv' | while read f; do echo "$f $(($(wc -l < "$f") - 1)) labels"; done
```

**Not built, on purpose:** the scorer that compares these labels with the system. It waits until Khurshid has approved
the targets above, so that they are fixed first.
