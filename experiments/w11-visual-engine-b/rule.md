# Visual Engine B selection rule (written 2026-11-12, before the grid and before v3 is trained)

**Goal.** At least 85 % frame accuracy on QiSens, blind. QiSens is never used to choose.

## Known before this rule (disclosed)

- **QiSens blind:**
  - B v1 67.26 (engine_a and dense candidates the same);
  - v2 selected 52.50, v1 selected 50.28, Engine A 50.00;
  - the other v2 and v1 rows of w10-qisens-engine-b-v2.
- **Public smoke checks of the tools** (not used to choose): the visual term at weight 0.5 to 2 on
  the process_480p30 clip (92.33 to 67.48 with B v1); 8 Kallax runs, untuned (Engine A 29.29,
  B v1 29.29, visual 26.12); one IKEA and one HA4M video with a one epoch v3 smoke model.
- **Intro detector:** w11-intro-detector-cv and -test.
- **50 Salads is not used:** its RGB is no longer downloadable (the Dundee host is gone, no
  archive copy).

## Selection sets (train parts only)

| set | videos | evidence | CV |
| --- | --- | --- | --- |
| HA4M macro phases | 152 train videos (cross subject train split), with the synthetic cuts p = 0.5 (ha4m_synthetic_cuts.json) | pose only (no RGB here): the visual term is inert | 3 folds by subject (ha4m_train_cv.json) |
| IKEA ASM Kallax Shelf Drawer | official training subset | pose (YOLO26 x runs) and visual | 3 folds by recording (ikea_asm_cv.json) |
| IKEA ASM Lack TV Bench | official training subset | pose and visual | the same |

The IKEA macro phases are in ikea_asm_macro_phases.md. Only the IKEA videos with a finished run
and visual.npz when the grid starts are used, and their list is stored with the result.

## Models

- **v1:** the w10-engine-b-v1 recipe.
  - HA4M: the CV models w10-engine-b-v1-cv.cvk.
  - IKEA: the shipped w10-engine-b-v1.pt (never trained on IKEA).
- **v2:** the same with w10-engine-b-v2-w4-c10.cvk on HA4M and w10-engine-b-v2.pt on IKEA.
- **v3:** the v2 recipe (contrast window 4 s, weight 1.0, action weight 1.0) on pose plus visual
  features, trained per fold k on the HA4M and IKEA train videos outside fold k
  (w11-engine-b-v3.cvk). Each held out video is decoded by the model that did not see it.

## Grid (dense candidates on)

| setting | values |
| --- | --- |
| model | v1, v2, v3 |
| `seg_a.visual_weight` | 0, 0.5, 1, 2 |
| `seg_a.visual_window_s` | 10, 20, 40 |
| `seg_a.embed_weight` | 0, 0.2, 1.0 |
| `seg_a.embed_window_s` | 10, 30 |
| `seg_a.engine_a_scale` | 1, 0.5, 0.25 |
| `seg_a.intro_detect` | off, on (k 1, sustain 20 s, share 0.5, from w11-intro-detector-cv) |

Refinement and video_norm stay off (w10-engine-b-v2 ablations, QiSens mixed).

## Criterion

- **Primary:** the worst case over the three sets of the CV mean frame accuracy.
- **Ties** (two decimals) go to the higher mean over the three sets, then to the simpler setting:
  intro off, visual weight 0, embed weight 0.2, Engine A scale 1, v1 before v2 before v3, the
  longer windows.
- **Reported:** each set separately, and the best row per model and per visual weight.

## Test parts, once

- **Rows** on the HA4M test split (p 0.5, also without cuts) and on the IKEA official testing
  subset of both furniture types:
  - Engine A;
  - B v1 as shipped (the default);
  - the selected setting;
  - the selected setting with intro detection switched to the other value.
- **v3 final:** w11-engine-b-v3.pt, trained on all train videos of both datasets.

## QiSens

- Khurshid runs the visual stage on the x run, then one compare.
- The pre chosen headline is the selected setting (v3 rows with the final v3 model).
- B v1 is the reference row.
- The default changes only with Khurshid's decision.
