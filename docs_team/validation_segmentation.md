# Segmentation validation report

**Status (2026-11-26).** Complete. All QiSens numbers are summaries that Khurshid ran and approved; nothing in this
file was read from the QiSens data area.

**Freeze.** All results below are for git tag `validation-freeze-1` (commit 866fba2). The tag fixes the code,
the configs, the phase descriptions and the SHA256 of the model weights
(docs_team/freeze/validation-freeze-1.json). The report tool records the tag and the commit it ran on.

**What these results are.** Development results on a single video. See section 5 before quoting any number.

## 1. Test set and hand labels

| item | value |
| --- | --- |
| videos | 1 (the QiSens maintenance video) |
| length | 596 s |
| frames | 17,892 |
| classes | 5 ordered phases (Removal, Disassembly, Cleaning, Adjustment and Inspection, Assembly and Installation) plus intro / other |
| annotators | 2 (Asel and Kazi), labelling independently; whole seconds; only the start times are boundaries |
| agreement | Cohen's kappa 0.980, frame agreement 98.49 %, mean start difference 1.80 s, largest 5.00 s (section 3) |
| final labels | steps_gt.csv: midpoint of the two annotators where they differ by 2 s or less |
| first boundary | set to 23 s (Asel) by the team lead, who checked it against the video |

The final labels are built from the two annotators, so a score against them is not independent of either
one. For that reason the system is also scored against **each annotator alone**. The agreement between the
two annotators is the **human agreement ceiling**: a system score above it would not be meaningful.

## 2. Runs and headline results

Approved summary numbers (2026-11-21). Frame accuracy in percent, MAE in seconds.

| label | run | pose | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engine B zero shot (flagship) | run_20261108_133251_1550 | YOLO26 x | 93.20 | 100 | 100 | 100 | 100 | 8.09 |
| Engine A (YOLO26 l) | run_20261121_051330_0511 | YOLO26 l | 75.02 | 100 | 100 | 100 | 80 | 29.35 |
| Engine B v1 | run_20261121_051432_2d75 | YOLO26 x | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
| Engine B v2 | run_20261121_051535_b39e | YOLO26 x | 52.50 | 100 | 80 | 60 | 20 | 35.51 |

For the three milestone copies, the values other than frame accuracy come from the earlier recorded runs of
the same settings (w10-qisens-l-replica, w10-qisens-engine-b-v1, w10-qisens-engine-b-v2); the report tool
recomputes them for the copies.

The flagship's stored scores use SigLIP at 2 samples per second and VLM windows of 2 s with 4 frames (the
setting of the 93.20). The default pipeline uses 1 sample per second and windows of 3 s with 2 frames: over 3
full runs it gave frame accuracy 92.97 to 93.40, median 93.40 (w11-qisens-zero-shot-timing).

## 3. Per phase results, confusion matrix, boundary errors

From tools/validate_seg.py, run by Khurshid on QiSens on 2026-11-26 at tag validation-freeze-1; approved summary
numbers. The full tables (confusion matrices, per boundary errors) stay in data/qisens/validation/.

**Human agreement (Asel against Kazi, the ceiling):** Cohen's kappa **0.980**, frame agreement **98.49 %**, mean
difference of the phase start times **1.80 s**, largest **5.00 s**.

**Frame accuracy against the final labels and against each annotator alone:**

| run | final labels | against Asel | against Kazi |
| --- | --- | --- | --- |
| Engine B zero shot (flagship) | 93.20 | 93.21 | 92.38 |
| Engine A (YOLO26 l) | 75.02 | 74.74 | 74.42 |
| Engine B v1 | 67.26 | - | - |
| Engine B v2 | 52.50 | - | - |

**Per phase F1 of the flagship (every video frame):**

| phase | F1 |
| --- | --- |
| intro / other | 23.08 |
| Removal | 86.36 |
| Disassembly | 88.60 |
| Cleaning | 93.33 |
| Adjustment and Inspection | 98.42 |
| Assembly and Installation | 99.79 |

**Interpretation.**
- The flagship is **about 5 points below the human agreement ceiling** (93.20 against 98.49 frame agreement), and it
  scores about the same against either annotator alone (93.21 and 92.38). So the result does not depend on whose
  labels are used.
- **Most of the remaining error is the intro:** 600 of the 690 intro frames (20 of 23 s) are predicted as Removal. The
  system starts the first phase at the start of the video instead of at 23 s. Without the intro the five work phases
  reach F1 86 to 100.
- **All other errors lie between neighbouring phases:** the confusion matrix has no frame given to a phase that is not
  next to the true one. That is expected from the ordered decoder, which can only move boundaries.
- Removal and Disassembly have the lowest F1 of the work phases (86 and 89); the later phases are near the ceiling.

**Robustness on the same video** (experiments/w12-robustness, rule fixed before the runs): weights at 0.5x and 2x, n = 8,
90.04 to 94.40 (median 93.20); random halves of the descriptions, n = 20, 90.55 to 94.40 (median 93.20); VLM sampling 3 s
and 2 frames, 93.40; pose YOLO26 l instead of x, 94.01. Every variant lies within 5 points of 93.20, so the result is
robust in every tested group. The independent descriptions of three team members (group c) are not tested yet.

These remain development results on a single video (section 5).

## 4. Metric definitions

**Frame accuracy.** The share of sampled frames with the correct label, intro / other included. The headline
uses the evaluation rate (eval.seg_fps = 15 frames per second); the report also gives it on every video frame.

**Edit score.** Normalised Levenshtein distance between the predicted and true sequences of segment labels,
as 100 x (1 minus distance / longer length). Intro / other is excluded.
**With this decoder the edit score carries almost no information:** the decoder is given the five phases in
their order and can only output them in that order, so whenever it finds all five the edit score is 100.

**Segmental F1@k (F1@10, F1@25, F1@50).** The metric of Lea et al. (CVPR 2017), as used by MS-TCN (Farha and
Gall, CVPR 2019); code in evaluation/metrics.py follows the MS-TCN implementation:

1. Take the predicted and true segments (maximal runs of one label), without intro / other.
2. For each predicted segment in time order, find the true segment of the **same label** with the highest
   intersection over union (IoU = overlap / span of the two).
3. It is a true positive if that IoU is at least the threshold k (0.10, 0.25 or 0.50) and that true segment is
   not matched yet. Otherwise it is a false positive.
4. True segments left unmatched are false negatives. F1 = 2 TP / (2 TP + FP + FN).

**Why F1 is coarse here.** The video has 5 true segments and the decoder predicts at most one per phase. So
F1 can only take the values 0, 20, 40, 60, 80 or 100 when all five are predicted: **one segment is worth 20
points.** A boundary moving by one second can flip F1@50 by 20 points, and F1@50 = 100 only says that each
phase overlaps its true segment by at least half. Different systems can share one F1 value while their
boundaries differ by tens of seconds. Frame accuracy, the per boundary errors and MAE are the more
informative numbers on this video.

**Boundary MAE.** For each true boundary, the distance to the nearest predicted boundary, averaged. It does not
penalise extra predicted boundaries.

## 5. Disclosure

These points limit what the results show and must accompany any quoted number.

1. **One video.** There is no second QiSens video. All results are on the same 596 s. There is no independent
   test set, so these are **development results on a single video**, not an estimate of performance on new
   videos. Robustness (experiments/w12-robustness) and the public train domain test
   (docs_team/train_domain_test.md) are the evidence beyond it.
2. **Phase descriptions.** docs_team/phase_descriptions.md was written and committed on 2026-11-12, before the
   first QiSens run of the zero shot stages, from the phase names and general railway maintenance knowledge,
   without any QiSens frame or annotation. It has not been edited since.
3. **The pre chosen setting was Z1 = 83.14.** Rule w11-zero-shot selected Z1 (SigLIP 1, crop 4, VLM 0.5) on public
   IKEA ASM data before any QiSens run. **83.14 is the pre registered result.**
4. **The default was changed after seeing the result.** Z2 (untuned equal weights 1, 1, 1) was a listed row of the
   rule and was not chosen by it. It gave **93.20** on QiSens and was made the default **after seeing both
   scores**. There was a reason independent of QiSens (the equal weights already had the better worst case on the
   public IKEA test, 57.79 against 50.43), but the decision was made with the QiSens numbers known, so **93.20 is
   optimistically biased** as an estimate. An honest range for this method on this video is 83.14 (pre chosen) to
   93.20 (selected after the fact).
5. **Several settings were scored on this video.** The rows in w11-qisens-zero-shot alone include Z1, Z2, their
   ablations, shot smoothing, the intro detector, the other class and the visual Engine B v3.
6. **Earlier engines were developed on the same video.** Engine A, Engine B v0, v1 and v2 were each evaluated on
   it during development, and what was learned shaped later choices, even though their weights were trained on
   public data only. The QiSens labels were never used to train a model.
7. **The decoder is given the task structure:** the list of five phases and their order. It does not discover
   them. The test says where the phases are, not what they are.
8. **The VLM carries most of the gain** (without it Z2 falls from 93.20 to 66.21), so the result depends on one
   model and on its prompt.

## 6. Commands (Khurshid runs them; nothing here opens the QiSens data for Claude)

```
cd ~/AssemblyVision/code && source ~/AssemblyVision/av.sh
python -m assemblyvision.tools.freeze --check --models
python -m assemblyvision.tools.validate_seg \
  --run ../data/qisens/runs/run_20261108_133251_1550 \
  --run ../data/qisens/runs/run_20261121_051330_0511 \
  --run ../data/qisens/runs/run_20261121_051432_2d75 \
  --run ../data/qisens/runs/run_20261121_051535_b39e \
  --gt ../data/qisens/annotations/steps_gt.csv \
  --annotator ../data/qisens/annotations/steps_raw_asel.csv \
  --annotator ../data/qisens/annotations/steps_raw_kazi.csv
```

CPU only, under a minute. Output: data/qisens/validation/validation_segmentation.{json,md}. The printed tables
hold summary numbers only (no frames, keypoints or boundary times). Paste them back once you approve them, and
they go into section 3.
