# Validation of the 10 company measurements

**Status (2026-11-26).** All 10 have an implementation, the targets are frozen (section 2) and the scorer is written
(section 4). **No measurement has been scored against hand labels yet:** the results come from the command in section 4,
which Khurshid runs on QiSens. Everything marked **experimental** stays so until it has ground truth.

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

**Targets frozen on 2026-11-26.** Khurshid approved the proposed targets 1 to 10 exactly as written below, before any
label was scored. They are not changed after a result is seen; a measurement that misses its target is reported as missing it.

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

## 3. Hand labelled ground truth (round 2, from 2026-11-28)

The first Labels page held the train domain clip classes, not measurement ground truth, so measurement labelling starts
with round 2. Items are drawn with a fixed seed from the frames where the worker is tracked, evenly over the phases (the
linked hand labels give the phases), so no system output decides what is labelled and every annotator gets the same
items. **Asel and Kazi each do all items**, so the kappa between them can be reported.

| page | task | items | measurements |
| --- | --- | --- | --- |
| Labels (team) | Frames: trunk angle from two clicks (hip midpoint, then shoulder midpoint; the page shows the angle and its band B0 under 20, B1 20 to 45, B2 45 to 60, B3 over 60), or cannot see; both hands working (yes, no, cannot see), gaze (towards the hands or the work, left, right, up, away, cannot see; left and right as seen in the image) | 100 frames | 3, 9, 10 |
| Labels (team) | Wrists: click the left and the right wrist, or mark it not visible | 50 frames | 6 (PCK@0.1) |
| Labels (team) | Events: in one 2 minute clip, mark each repeated cycle start, tool pickup and object placement | 1 clip | 4, 7, 8 |
| Ratings (mentor only) | The run's suggestions with their evidence: useful, partly useful or not useful, optional comment | 4 on the flagship | 5 |

**The 2 minute clip** is chosen by a fixed rule, without anyone looking at the video: among the phases of at least 2
minutes, the one where the wrists move most (median wrist speed relative to the hip); its middle 2 minutes.

**Trunk (measure 3), changed on 2026-11-28 before any trunk label was scored:** the band radio buttons were replaced by
two clicks, because nobody can judge 20, 45 or 60 degrees by eye. **The system's trunk angle is 2D:** the line from the
mid hip to the mid shoulder of the cleaned image keypoints against the upward image vertical (analysis/motion.trunk_angle,
atan2(|dx|, -dy)), not the 3D body. The clicked angle is computed with that same function, so both are image angles and
the comparison is like for like. **Limit:** an image angle underestimates bending toward or away from the camera, for the
person and the system alike, so the score says how well the system finds the keypoints, not how well it measures the true
3D bending. Scored: exact band agreement and within one band (targets unchanged), the mean absolute angle difference in
degrees, and the hip and shoulder point errors in pixels, as a share of the torso length and as the share within 0.1 of
the torso length. The band answers of the first version are not used (tools/migrate_frame_labels.py drops them from saved
files and keeps the hands and gaze answers).

**Rules of the comparison (fixed before any label is scored):** both hands
working = both wrists active in at least half of the frames within 1 s; gaze: the system's "down" counts as towards the
hands or the work, "front" (facing the camera) and the label "up" have no counterpart and count as misses; wrist PCK@0.1 =
within 0.1 of the longer side of the worker's tracked box; repetitive work in 10 s windows of the clip, where a window
with at least two cycle starts counts as repetitive (**limit:** the system finds short repeated motions of 0.4 to 2 s,
while the annotators mark work cycles, so the two do not measure quite the same thing); events matched one to one within
1 s; suggestions: the share rated useful. Hand travel can only be scored for its PCK part; the travel itself needs
consecutive frames.

**Access.** Labels is shown to the team's accounts and hidden from the mentor; Ratings is shown only to the mentor. Every
save checks the account on the server, so the page address alone does not open a task. Labels are saved per annotator in
`data/qisens/validation/labels/<run>/<task>/<user>.csv`. The data rules of docs_team/labelling_guide.md section 1 apply:
no screenshots, no downloads, and clear the browser cache after a QiSens session.

## 4. Scores against the hand labels

**Scorer:** tools/score_measurements.py (written after the targets were frozen). For every labelled item it computes the
system's own answer from the run's stored keypoints with the same code and settings as the metrics (the rules are in the
tool's docstring), compares it with the label, and judges the result against the frozen target. Each annotator is scored
alone; here that is **one annotator, Khurshid**, so no human agreement is known for these tasks. Items marked
"cannot tell" by the annotator are left out; items where the system has no answer (for example no trunk measured at that
frame) are counted as not covered and shown, not dropped silently.

**Command (Khurshid runs it; CPU, prints summary numbers only):**

```
cd ~/AssemblyVision/code && source ~/AssemblyVision/av.sh
python -m assemblyvision.tools.detect_objects --run ../data/qisens/runs/run_20261108_133251_1550
python -m assemblyvision.tools.score_measurements --run ../data/qisens/runs/run_20261108_133251_1550 --gt ../data/qisens/annotations/steps_gt.csv
```

The first command is only needed for 7 and 8 (objects.npz exists already from round 1). The same score command reads the
round 2 labels; a task nobody has labelled prints "no labels yet".

**Results, round 1** (run by Khurshid on QiSens on 2026-11-28, recorded unchanged). At that time only the phase labels
existed; the earlier Labels page had held the train domain clip classes, not measurement ground truth.

| # | measurement | annotator | result | target | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | process time | two annotators (final labels) | boundary MAE 8.093 s; 4 of 5 phase durations within 10 % | mean boundary error at most 10 s, each phase duration within 10 % | **misses the target** (the frozen target needs all 5 phases) |
| 2 | movement path | - | **not validated on QiSens:** no walk of known length exists in the video | within 20 % over a walk | - |
| 3 | waist bending | - | no labels yet | exact band at least 80 %, within one band at least 95 % | - |
| 4 | repetitive work | - | no labels yet | F1 at least 0.70 | - |
| 5 | improvable motions | - | no labels yet | at least 60 % rated useful | - |
| 6 | hand travel | - | no labels yet | wrist within PCK@0.1 for at least 85 %; travel within 20 % | - |
| 7 | tool pickup time | - | no labels yet | events within 1 s: recall and precision at least 0.70 | - |
| 8 | object placement time | - | no labels yet | events within 1 s: recall and precision at least 0.70 | - |
| 9 | two hand cooperation | - | no labels yet | agreement at least 75 % | - |
| 10 | gaze direction | - | no labels yet | agreement at least 70 % | - |

The flagship's movement path per phase (GVHMR floor, metres, **not validated**): Removal 23.82, Disassembly 6.04,
Cleaning 12.74, Adjustment and Inspection 6.56, Assembly and Installation 33.6.

**Limit of measure 2.** The path cannot be validated on the QiSens video: it has no walk between points of known
distance, and the floor scale of GVHMR is estimated from the body. A validation needs a new short recording with a taped
distance; until then the path is shown as not validated.

A task Khurshid did not label prints as "no labels yet" in place of a number.
