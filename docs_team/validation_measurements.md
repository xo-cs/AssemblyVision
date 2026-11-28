# Validation of the 10 company measurements

**Status (2026-11-28).** All 10 have an implementation, the targets are frozen (section 2) and the scorer is written
(section 4). The hand labelling is reduced: **one annotator (Khurshid)** for every task; #3 (trunk) and #6 (wrists) are
validated automatically on the public COCO 2017 val keypoints (experiments/w12-coco-pose), which is not factory footage;
#2 (movement path) is not validated (future work). **No measurement has been scored against hand labels yet:** the results come from the command in section 4,
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
| 2 | movement path | mid hip on the GVHMR floor at 5 Hz, summed per shot part | a walk between points of known distance | m | within 20 % over a walk | **not validated:** the QiSens video has no walk of known distance; a recording with a taped distance is future work | each shot part has its own world frame, so walks across parts are not joined; jitter adds length; GVHMR scale is estimated |
| 3 | waist bending | 2D trunk angle from the mid shoulder and mid hip keypoints against the image vertical, in 4 bands | the COCO 2017 val human keypoints of the same four joints (public, automatic) | band | exact band at least 80 %, within one band at least 95 % | **meets the target on COCO:** exact band 0.955, within one band 0.999, mean difference 2.58 deg (2,800 persons); bent subset over 45 deg: exact band 0.747, within one 1.0, 7.0 deg (150 persons) | COCO is not factory footage (mostly upright people in everyday photos); 2D: bending toward the camera is underestimated; the bent subset is below the exact band target |
| 4 | repetitive work | autocorrelation over 10 s windows | a person says repetitive or not for 10 s clips | yes/no | F1 at least 0.70 | not measured | misses cycles longer than 2 s |
| 5 | improvable motions (**experimental**) | rule list on the measured metrics | an industrial engineer rates each suggestion as useful or not | share useful | at least 60 % rated useful | not measured | a judgement; the thresholds are defaults; inherits every limit of the metrics it uses |
| 6 | hand travel | wrist path relative to the hip | the COCO 2017 val human wrist keypoints (public, automatic) | torso lengths | wrist within PCK@0.1 for at least 85 %; travel within 20 % | **PCK part met on COCO:** wrist PCK@0.1 0.935 (left 0.934, right 0.936); the travel part is not measurable from single images | COCO is not factory footage; occluded hands |
| 7 | tool pickup time (**experimental**) | wrist speed drops at a detected object (YOLO26 m, COCO, 2 per second), and the object then moves with the hand; time = the reach | a person marks each moment a hand closes on a tool | s | events within 1 s: recall and precision at least 0.70 | not measured | COCO has almost no workshop tools, so most pickups are missed; 2D distance; occlusion |
| 8 | object placement time (**experimental**) | wrist speed drops at a detected object, and the hand then leaves while the object stays; time = the reach | a person marks each moment a hand lets go of a placed object | s | events within 1 s: recall and precision at least 0.70 | not measured | as for 7; a hand merely passing a static object can count |
| 9 | two hand cooperation | both wrists above a speed threshold | one annotator (Khurshid) says whether both hands are working, on 30 frames | yes/no | agreement at least 75 % | pending the 30 frame labels | one annotator, no human agreement; fast idle motion counts as work; a holding hand counts as idle |
| 10 | gaze direction | yaw from nose, eyes and ears | one annotator (Khurshid) assigns the direction on 30 frames | class | agreement at least 70 % | pending the 30 frame labels | one annotator; head, not eyes; poor when the face is small or turned away |

## 3. Ground truth (from 2026-11-28)

**One annotator.** Every hand labelling task is done by Khurshid alone, so no human agreement (kappa) is reported for
the measurements; the accounts of Asel, Kazi and Huzaifa were removed. Items are drawn with a fixed seed from frames with a
tracked worker, evenly over the phases, so no system output decides what is labelled.

| source | task | items | measurements |
| --- | --- | --- | --- |
| COCO 2017 val (public, automatic) | human keypoints of shoulders, hips and wrists against YOLO26 x with the flagship's settings | 2,878 persons with the four torso points labelled visible | 3, 6 |
| Labels page (team) | Frames: are both hands working (yes, no, cannot see), gaze (towards the hands or the work, left, right, up, away, cannot see; left and right as seen in the image) | 30 frames (a subset of the earlier 100 frame draw, 6 per phase, so earlier answers stay valid) | 9, 10 |
| Labels page (team) | Events: in one 2 minute clip, each repeated cycle start, tool pickup and object placement | 1 clip | 4, 7, 8 |
| Ratings page (mentor only) | The run's suggestions with their evidence: useful, partly useful or not useful | 4 on the flagship | 5 |
| none | a walk of known distance | - | 2 (not validated, future work) |

**Both hands working** means yes only if both hands do the task (holding, turning, guiding a part or tool), even different
things; a hand that only braces the body or rests counts as no (the page says so).

**Why COCO for #3 and #6.** A person cannot judge a trunk angle by eye, and clicking keypoints on QiSens frames by hand is
slow; COCO has human keypoints for thousands of people. The trunk angle is computed with the system's own function
(analysis/motion.trunk_angle, 2D) on both the human and the predicted keypoints. **Limit:** COCO shows people in everyday
photos, mostly upright and from many views, not a worker bending over a bogie; so the result says how well the keypoints are
found in general, not on QiSens. On the bent subset (human angle over 45 degrees) the exact band is 0.747, below the target.

**The 2 minute clip** is chosen by a fixed rule, without anyone looking at the video: among the phases of at least 2
minutes, the one where the wrists move most (median wrist speed relative to the hip); its middle 2 minutes.

**Rules of the comparison (fixed before any label is scored):** both hands working = both wrists active in at least half of
the frames within 1 s; gaze: the system's "down" counts as towards the hands or the work, "front" (facing the camera) and the
label "up" have no counterpart and count as misses; repetitive work in 10 s windows of the clip, where a window with at
least two cycle starts counts as repetitive (**limit:** the system finds short repeated motions of 0.4 to 2 s, the annotator
marks work cycles); events matched one to one within 1 s; suggestions: the share rated useful.

**Access.** Labels is shown to the team's accounts and hidden from the mentor; Ratings only to the mentor; every save
checks the account. Labels are saved in `data/qisens/validation/labels/<run>/<task>/<user>.csv`. The data rules of
docs_team/labelling_guide.md section 1 apply.

## 4. Scores against the hand labels

**Scorer:** tools/score_measurements.py (written after the targets were frozen). For every labelled item it computes the
system's own answer from the run's stored keypoints with the same code and settings as the metrics (the rules are in the
tool's docstring), compares it with the label, and judges the result against the frozen target. The labels come from
**one annotator, Khurshid**, so no human agreement is known for these tasks; #3 and #6 come from COCO. Items marked
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
