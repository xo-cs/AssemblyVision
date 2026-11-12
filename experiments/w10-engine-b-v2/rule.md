# Engine B v2 selection rule

Written and committed 2026-11-12, before any v2 training or decoder grid. No result of the
benchmark test split had been seen for any v2 setting, decoder option or CV model at that point.
The only test split result known was the baseline w10-phase-bench-baseline: Engine A 24.01 and
Engine B v1 24.01 mean frame accuracy. A diagnostic on 20 test videos (raw boundary head peaks as
candidates gave 51.2) showed that the v1 decoder has no candidates on HA4M. That is why the decoder
grid below includes `seg_a.embed_candidates`. That diagnostic is disclosed here and is not used
for choosing.

## Data

- **Selection set:** the public phase benchmark (experiments/datasets/ha4m_macro_phases.md).
- **Train split:** the 152 cross subject train videos (28 subjects). It is split into 3 folds by
  subject (`experiments/datasets/ha4m_train_cv.json`, seed 42, stratified by lab), and every
  setting is scored by cross validation: each fold's videos by the model trained on the other two
  folds.
- **Test split:** the 65 cross subject test videos (13 subjects). They are scored once, at the
  end, for the rows named in the last section. QiSens is never used for choosing.

## Candidates

**Models.** Each is trained on each CV fold, and the winner is trained once more on all 152 train
videos:

- `v1`: the v1 recipe (w10-engine-b-v1 settings), retrained on the CV folds so it is scored like
  the others. Its shipped model, trained on all 152 train videos, is the test split v1.
- `v2`: the v1 network plus a supervised contrastive objective on window embeddings. The windows
  are the mean last stage features over non overlapping windows of `engine_b.contrast_window_s`.
  A window counts only if it lies inside one macro phase. Windows of the same macro phase (across
  the videos of a batch) are positives, all others negatives, with temperature
  `engine_b.contrast_temp = 0.1`. Grid: `contrast_window_s` in {4, 8} s and
  `contrast_weight` in {0.5, 1.0}, which gives 4 v2 models. Everything else is the v1 recipe.

**Decoder settings,** for every model:

| setting | values |
| --- | --- |
| candidates (amendment 1) | `engine_a`, `peaks` (`seg_a.embed_candidates` on), `dense` (`engine_b.candidates`) |
| `seg_a.embed_weight` | 0.2, 0.4, 0.8 |
| `seg_a.embed_window_s` | 10, 20, 30 |
| `seg_a.refine_window_s` (item 6) | 0 (off), 5, 10 |
| `engine_b.video_norm` (item 7) | off, on |

That is 162 decoder settings per model and 810 in all (108 and 540 before amendment 1).

## Criterion

- **Primary:** the mean frame accuracy over the 152 train videos under cross validation.
- **Ties** (equal at two decimals) are broken by, in order: higher mean F1@50; lower mean MAE; the
  simpler setting (options off before on, candidates in the order engine_a, peaks, dense, v1 before v2, the smaller weight, then the longer window).
- **Chosen:**
  - **v1 selected:** the best decoder setting for the v1 model;
  - **v2 selected:** the best model and decoder setting among the v2 models.

## Test split, once

- **Rows:** Engine A; Engine B v1 as shipped (the current default, dense candidates since
  amendment 1); v1 selected; v2 selected.
- **Ablations:** for v1 selected and v2 selected, each with refinement switched to the other value
  and with video_norm switched to the other value. These are information on items 6 and 7, never
  used for choosing.
- **Decision:**
  - v2 replaces v1 only if v2 selected has a higher mean frame accuracy than v1 selected on the
    test split;
  - a decoder option (candidates, refinement, video_norm) becomes part of the default only as
    part of the selected configuration of the model that is used;
  - if v1 selected beats v1 as shipped, it becomes the default "B v1" decoder only with
    Khurshid's approval, because it changes the blind QiSens setting.
- **QiSens:** the pre chosen headline for the QiSens compare is the configuration chosen here.

## Item 8 (intro and outro), separate

- `seg_a.other_ends` and its bias are evaluated on synthetic intros built from the benchmark
  (still segments of 5 to 30 s prepended).
- The bias is chosen on the train split, and the result is reported on the test split once.
- It is not part of the selection above, and it stays off by default unless Khurshid decides
  otherwise.

## Amendment 1 (2026-11-12, before the decoder grid)

Khurshid stopped the work before the CV grid: with no candidates on HA4M the benchmark cannot
separate any variant. A new decoder option, `engine_b.candidates` = `engine_a` (only Engine A's
candidates) or `dense` (their union with a grid every `engine_b.candidate_step_s` = 1 s, strength
from the change score, plus the boundary head when refinement is on), is now part of the "B v1"
preset. The candidate setting is a grid dimension (table above). The step stays at 1 s, and
everything else in this rule is unchanged.

Disclosed: before this amendment, the five CV models had been trained (training logs only, no
decoder scores), and the test split baseline rows were run once, as item 3 asks for every new
baseline (experiments/w10-phase-bench-baseline-dense):

- Engine A: 24.01;
- B v1 with engine_a candidates: 24.01;
- B v1 with dense candidates: 51.42.

No train fold decoder score had been seen. The baseline rows are not used for choosing.

## Amendment 2 (2026-11-12, before the decoder grid)

Khurshid's changes to the grid (the first grid run was stopped before it wrote anything):

**Candidates.** Dense candidates on for every setting (`engine_b.candidates = "dense"`). The
candidate dimension of amendment 1 is dropped.

**Scales.** The embed weight 0.2 was chosen when Engine B could only reweight Engine A's
candidates. New dimensions:

| setting | values |
| --- | --- |
| `seg_a.embed_weight` | 0.2, 0.5, 1.0, 2.0 |
| `seg_a.engine_a_scale` (multiplier on the base strength of every Engine A candidate) | 1.0, 0.5, 0.25, 0 (0 = Engine B alone) |
| `seg_a.embed_window_s` | 10, 20, 30 |
| `seg_a.refine_window_s` | 0, 5, 10 |
| `engine_b.video_norm` | off, on |

That is 288 decoder settings per model and 1440 in all. Only the ratio of the two scales changes
the DP's choice (every strength is linear in them). So for example (embed 0.5, scale 0.25)
decodes like (embed 2.0, scale 1.0), and with scale 0 every embed weight gives the same result.
Equal settings tie and fall to the simplicity order.

**Synthetic editing layer** (benchmark, train and test splits alike):

- Cut candidates at the cut rate of the QiSens video. **Disclosed: the rate comes from the QiSens
  shot count** (37 shots in 596 s, so 36 cuts, 3.62 cuts per minute). The shot count is the only
  QiSens figure used, and it is not a label.
- A share p (0.5, 0.75, 1.0) of the true macro phase boundaries of a video get a cut on the
  boundary (chosen at random). The other cuts of that video fall at random frames.
- Cuts per video = round(rate × duration). Seed 42, per (video, p).
- Stored once in `experiments/datasets/ha4m_synthetic_cuts.json` (217 videos; on average 6.1
  cuts and 4.0 phase boundaries per video).
- Every setting is scored without cuts and with the cuts of each p.

**Criterion** (replaces the primary criterion above, for every dimension including the model):

- the worst case over the three p values of the mean frame accuracy over the 152 train videos
  under cross validation;
- ties (two decimals) are broken by higher F1@50, then lower MAE, both in that worst condition,
  then by the simpler setting: refinement off, video_norm off, Engine A scale 1 first (then 0.5,
  0.25, 0), v1 before v2, the smaller embed weight, the longer window.
- Every p and the no cut condition are reported separately.

**Test split, once:** the rows of the section above, each without cuts and with each p.

**Decoder fix found before the grid** (disclosed):

- A smoke check of the scale dimension on 4, then 8, train videos of fold cv1 (v1 CV model;
  train data only) showed dense candidates near the start of the active span. There the change
  score window is cut short and noisy: a boundary was chosen at 7 s.
- Fix: no dense candidate closer than half of `seg_a.embed_window_s` to either end of the active
  span. Engine A's candidates are unchanged.
- The test split baseline 51.42 of amendment 1 (B v1 dense) was measured before this fix. The
  "B v1 as shipped" row of the final test measures it again with the fix.
