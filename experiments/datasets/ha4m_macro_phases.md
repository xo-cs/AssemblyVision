# HA4M macro phases (public phase benchmark, Week 10)

Written 2026-11-12, before any Engine B v2 training. Code: `code/assemblyvision/datasets/ha4m_phases.py`.

## Grouping

| phase | actions (HA4M Table 3) | mean s | median s | min s | max s |
| --- | --- | --- | --- | --- | --- |
| Block 1 | 1 carrier, 2 gear bearings (x3), 3 planet gears (x3), 4 carrier shaft | 21.8 | 21.1 | 7.0 | 42.9 |
| Block 2 | 5 sun shaft, 6 sun gear, 7 sun gear bearing, 8 ring bear | 15.4 | 14.8 | 3.3 | 29.6 |
| Join blocks | 9 pick up Block 2 and place it on Block 1 | 23.7 | 17.2 | 3.8 | 166.1 |
| Cover and screws | 10 cover, 11 screws (x2) | 19.6 | 17.6 | 5.9 | 58.9 |
| Fasten | 12 Allen key, turn both screws, return the key and the EGT | 14.8 | 13.0 | 4.5 | 49.6 |

The durations are over all 217 videos. A video lasts 97 s at the median (33 to 242 s).

## Rationale

- **Task order from the paper.** Cicirelli et al. (Scientific Data 2022, Table 3 and its note)
  describe the Epicyclic Gear Train assembly in three phases: "First, the assembly of Block 1
  (action IDs 1 to 4), then Block 2 (action IDs 5 to 8) and finally the EGT (action IDs 9 to 12)."
  The first two macro phases are exactly those blocks.
- **The final EGT phase is split along its own sub tasks.** As one phase it would last 58 s on
  average, about as long as both blocks together, and its three parts are different work: carrying
  and seating Block 2 on Block 1 (9), placing parts (10 cover, 11 screws), and a tool phase
  (12 Allen key).
- **This gives the target shape.** There are 5 phases per video (the QiSens process also has 5),
  each tens of seconds (medians 13 to 21 s), in a fixed task order. Like the QiSens phases, each
  one is a sequence of several short actions, not a single gesture.
- **Alternatives rejected:**
  - 3 phases (the paper's split): the last phase is too long and too mixed;
  - 4 phases (9 to 11 together): it puts an action of carrying a whole block into a placing phase;
  - 6 phases (10 and 11 apart): cover and screws are 7 s and 12.6 s on average, too short for
    "tens of seconds".

## GT rules (per frame, from Labels.txt)

- **Each frame takes the macro phase of its action.**
- **Don't care frames (action 0):**
  - between two frames of the same phase, they belong to that phase;
  - between two different phases, they are split at the midpoint;
  - before the first and after the last action, they are `other` (intro and outro; mean 1.0 s
    and 1.8 s, at most 5.9 s).
- **Out of order videos:** a video whose actions leave the task order is kept, and its phases then
  appear in another order or twice. This happens in 5 of 217 videos, 2 of the 65 test videos (one
  of the 217 has 9 segments, two have 6). The engines always get the 5 names in task order, as they
  get the steps.toml order on QiSens.
- **The repetition index is not used:** repetitions stay inside one phase.

## Benchmark protocol

- **Videos:** the cross subject fold of `ha4m_splits.json`, 152 train videos (28 subjects) and 65
  test videos (13 subjects). Engine B v1 was trained on exactly these 152 train videos, so the
  test videos are unseen.
- **Input:** each video's Kinect COCO 17 skeleton from the Engine B cache, with YOLO like
  confidences and no augmentation, as a one shot timeline with one subject. Both engines decode it
  as they decode a run.
- **Metrics:** the QiSens metrics per video (frame accuracy, edit, F1@10/25/50, boundary MAE),
  then the mean over videos. The table also gives the median frame accuracy and the number of
  videos at 85 or above.
- **Public clips:** the two clips with a GT (process_480p30, shots_check_480p30) are scored from
  their stored YOLO26 x runs (w10-pose-sens-x).
- **Use:** the train split is for choosing settings (cross validation by subject); the test split
  is evaluated once at the end (see experiments/w10-engine-b-v2/rule.md).

## Known limits

- The benchmark keypoints are Kinect joints made to look like YOLO output, not YOLO26 on RGB.
  The RGB route was dropped, and the domain gap is measured in w10-ha4m-domain-gap and
  w10-ha4m-domain-gap-l.
- HA4M has one fixed camera, and the subject is always in view and almost always working. There
  are no cuts and few pauses, so Engine A's pause and cut evidence is nearly absent here, unlike
  QiSens.
