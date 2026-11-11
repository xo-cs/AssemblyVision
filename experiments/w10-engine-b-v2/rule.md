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
| `seg_a.embed_candidates` | off, on |
| `seg_a.embed_weight` | 0.2, 0.4, 0.8 |
| `seg_a.embed_window_s` | 10, 20, 30 |
| `seg_a.refine_window_s` (item 6) | 0 (off), 5, 10 |
| `engine_b.video_norm` (item 7) | off, on |

That is 108 decoder settings per model and 540 in all.

## Criterion

- **Primary:** the mean frame accuracy over the 152 train videos under cross validation.
- **Ties** (equal at two decimals) are broken by, in order: higher mean F1@50; lower mean MAE; the
  simpler setting (options off before on, v1 before v2, the smaller weight, then the longer window).
- **Chosen:**
  - **v1 selected:** the best decoder setting for the v1 model;
  - **v2 selected:** the best model and decoder setting among the v2 models.

## Test split, once

- **Rows:** Engine A; Engine B v1 as shipped (the current default); v1 selected; v2 selected.
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
