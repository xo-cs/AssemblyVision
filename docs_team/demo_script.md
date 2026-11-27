# Demo script: the flagship run in 10 minutes

**Run:** run_20261108_133251_1550 (QiSens maintenance video, 596 s, Engine B zero shot, YOLO26 x).
**Open:** http://192.168.0.171:7860/?view=results&run=run_20261108_133251_1550 and log in.
In the Results list it is "Engine B zero shot (flagship) · 93.2 %".

**Before the meeting (about 15 minutes earlier):**
1. If the app was restarted, wait 2 minutes: the keep warm loop fills the flagship's views, so it opens in about 1 s.
2. Open the run once and click through the steps below, so nothing loads for the first time in front of the audience.
3. For the tool pickup numbers, run detect_objects on the flagship once (CPU), see docs_team/demo_outputs.md.
4. Have docs_team/validation_segmentation.md and experiments/w12-train-domain/table.md open in a second window for part B.

Every step was checked on 2026-11-27 in a test copy of the app (the same code) on a public run with 3D and 4D: the
timeline, the video, the floor maps (22 parts), the cycle chart, the suggestions, the export (4 files) and the 3D and 4D
players work, with no error in the page or the app log. The flagship itself was not opened by Claude (QiSens rule); the
numbers below are the approved ones.

## Part A: what the system gives (about 6 minutes)

| # | min | what to click | what to say | number to show |
| --- | --- | --- | --- | --- |
| 1 | 0:00 | Results, the flagship in the list; scroll to **Process timeline** | "From one video the system finds the five maintenance phases in order and when each one happens. No QiSens footage was used to train or tune it." | 5 of 5 phases; frame accuracy **93.2 %** (card at the top) |
| 2 | 1:00 | **Overlay video**: play it for 10 seconds; optionally **Comparison mode** | "The skeleton of the worker is tracked through the whole video; S is the worker the analysis follows, and the current phase is in the header." | the phase label changing in the header |
| 3 | 2:00 | **Movement**, 2D: the hand reach heatmap, then open **Floor map per shot part (3D body, metres)**; then **Movement path on the floor** further down | "Where the hands work and where the feet stood, in metres on the floor, from a 3D body model. The path in metres per phase is not validated yet." | the path in metres of the longest phase (read off the table) |
| 4 | 3:00 | **Cycle times** (the bar chart) and **Cycle baseline** | "The duration of every phase. With one pass through the process per video, each phase has one cycle; with repeated cycles the chart shows one bar per cycle." | the longest and the shortest phase in seconds (read off the bars) |
| 5 | 4:00 | **Improvement suggestions** table, then **Report**, **Export report**, open report.pdf | "The suggestions are rule based, each with its evidence: the metric, its value, the threshold and the time ranges. They are suggestions for a person to check, not findings. The PDF has the timeline, the measured baseline, the cycle times and these suggestions." | the number of suggestions; one row with its time range |

## Part B: why we trust it (about 3 minutes)

| # | min | where | what to say | number to show |
| --- | --- | --- | --- | --- |
| 6 | 5:30 | **Evaluation** card on the page; validation_segmentation.md, section 3 | "Against two independent annotators the flagship is 93.20 % frame accuracy, about 5 points below the human agreement ceiling. Most of the remaining error is the 23 s intro." | 93.20; per phase F1: Removal 86.36, Disassembly 88.60, Cleaning 93.33, Adjustment and Inspection 98.42, Assembly and Installation 99.79; intro 23.08 |
| 7 | 6:30 | validation_segmentation.md, sections 1 and 3 | "The two annotators agree almost completely, and the system scores the same against either of them, so the result does not depend on whose labels we use." | kappa **0.980**, frame agreement **98.49 %**; system vs Asel 93.21, vs Kazi 92.38 |
| 8 | 7:00 | experiments/w12-robustness/table.md | "We changed the weights, half of the descriptions, the VLM sampling and the pose model. Every variant stayed within 5 points of 93.20." | weights 90.04 to 94.40; description halves 90.55 to 94.40; VLM 93.40; pose l 94.01; group c pending |
| 9 | 7:45 | experiments/w12-train-domain/table.md | "On 14 public railway clips it has never seen, with other classes and no tuning, it named 13 of 14 correctly." | **13 of 14 (92.86 %)**, chance 16.7 %; joined video 89.69 |
| 10 | 8:30 | validation_segmentation.md, section 5 | "Honestly: this is one QiSens video, the default was chosen after both results were known (the pre chosen setting gave 83.14), and the earlier engines were developed on the same video. So these are development results." | 83.14 pre chosen, 93.20 selected afterwards |

## Part C: optional views (about 1 minute)

| # | min | what to click | what to say | number to show |
| --- | --- | --- | --- | --- |
| 11 | 9:00 | **Movement**, **3D**: play | "The worker's skeleton in 3D, in metres, with hand and foot positions building up over time." | none |
| 12 | 9:30 | **Movement**, **4D**: play, drag to orbit | "The scene as points with the person, the camera path and the skeleton. These views are optional; no step or metric depends on them." | none |

**If something does not load:** press **Refresh** next to the run list and wait 2 seconds. The 4D view loads one shot
part at a time (about 4 MB), so give it a moment on a slow network.

**Do not claim:** a validated movement path, tool pickup counts (the detector does not know workshop tools; a 0 means not
seen), or a general accuracy on new videos.
