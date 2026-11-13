# Other class decision rule (written 2026-11-13, before any run with seg_a.other_class)

**Question.** Should the zero shot scorers get an extra class "no work in progress", allowed only as an
intro before the first phase and an outro after the last (seg_a.other_class), on or off by default?
QiSens is never used to decide.

## Known before this rule (disclosed)

- **QiSens blind:** "B zero shot" (untuned equal weights) 93.20; intro_detect on it changes nothing
  (w11-qisens-zero-shot). Nothing is known about where the remaining QiSens errors are.
- **IKEA ASM GT:** the frames outside the annotated actions are few: at the start a median of 2.1 to
  2.8 s, at the end 1.5 to 3.5 s, 4 to 10 % of the frames (measured on the GT only, before this rule).
- **The prompt with the other class** keeps the frozen phase lines and adds one answer after them
  ("Before the work starts or after it has ended, the frames can also show: ..."). The frozen JSON
  template has a doubled quote after the last name; it is kept, so the other class prompt has it too.
- No score of the other class has been computed.

## Fixed parts

- **Descriptions:** docs_team/phase_descriptions_other.md (committed with this rule): one group for
  QiSens, one for IKEA ASM. The phase description files stay unchanged.
- **Decoder:** the untuned "B zero shot" weights (siglip 1, crop 1, vlm 1, temperature 0.01, shot_smooth
  off, B v1 boundary evidence, dense candidates), the synthetic cuts p 0.5.
- **Other class settings, not tuned:** seg_a.other_class_max_share 0.3 (the longest intro or outro, a
  share of the span with a person) and seg_a.other_min_strength 0.15 (the existing other_ends value).
- **VLM:** the setting chosen by rule w11-vlm-speed. Off reads its file, on reads the same setting
  scored with the other class prompt (the _other file).

## Videos and conditions

The 40 IKEA ASM training videos of w11-vlm-speed (experiments/w11-vlm-speed/videos.json, 20 Kallax Shelf
Drawer, 20 Lack Coffee Table).

1. **Natural:** the videos as they are.
2. **Synthetic intro and outro:** before and after each video, "no work" material from the frames
   outside the annotated actions of the other videos of the same furniture type, spliced at the level of
   every per frame evidence (person, motion, the B v1 embedding, the phase scores), with a cut at every
   junction. Intro 20 to 60 s, outro 10 to 30 s, seeded (seed 42) per video. The spliced frames are
   "other" in the GT (tools/other_class_bench.py).

## Criterion

- **Measure:** the worst case over the two sets of the mean frame accuracy.
- **On** if both hold:
  - natural: the worst case drops by at most 1.0 point with other_class on;
  - synthetic: the worst case rises by at least 2.0 points.
- **Off** otherwise (the default stays off).
- **Reported:** both conditions per set, F1@50 and boundary MAE.

## QiSens

Khurshid runs the stages that add the other class on the x run, then one compare,
seg_a.other_class false,true on the "B zero shot" setting. The default changes only with his decision.
