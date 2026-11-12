# Zero shot phase scoring selection rule (written 2026-11-12, before any grid or result of it)

**Goal.** Above 85 % (aim 90) frame accuracy on QiSens, blind. No public dataset shares the QiSens
phases, so the phases are scored zero shot by meaning (docs_team/phase_descriptions.md) with
SigLIP 2 and Qwen3 VL. Only the few weights of the decoder are chosen on public data.

## Known before this rule (disclosed)

- **QiSens blind:** B v1 67.26, v2 52.50, Engine A 50.00 and the rows of w10-qisens-engine-b-v2.
- **Smoke checks on public data:**
  - one Kallax video: the argmax of the SigLIP full frame scores agrees with the GT macro phase on
    51.4 % of its samples, the crop on 48.6 %;
  - a unit test of the likelihood term.
- **Not used for choosing:** the IKEA description file was frozen before both.

## Sets (IKEA ASM; HA4M has no RGB here)

- **Kallax Shelf Drawer** (4 phases) and **Lack Coffee Table** (3 phases), with the descriptions of
  docs_team/phase_descriptions_ikea_asm.md.
- **Choice:** on the official training subset. The decoder has no trained part here, so no CV is
  needed: every training video is scored as is.
- **Test:** the official testing subset, once.
- **Editing layer:** every video is scored with the synthetic cuts of IKEA (ikea_bench.make_cuts,
  the QiSens cut rate, p = 0.5: half of the true boundaries get a cut, as for HA4M). IKEA videos
  are single shots otherwise, so shot_smooth could not be judged on them. Also reported without
  cuts.

## Fixed parts

- The boundary evidence of B v1: the v1 pose embedding, weight 0.2, window 30 s.
- Engine A's candidates, plus dense candidates, so the likelihood can place a boundary anywhere.
- The visual term and the intro detector stay off. Those belong to rule w11-visual-engine-b, and
  their combination comes after both rules are done.

## Grid

| setting | values |
| --- | --- |
| `seg_a.siglip_weight` (full frame) | 0, 0.5, 1, 2, 4 |
| `seg_a.siglip_crop_weight` | 0, 0.5, 1, 2, 4 |
| `seg_a.vlm_weight` | 0, 0.5, 1, 2, 4 |
| `seg_a.siglip_temp` | 0.01, 0.03 |
| `seg_a.shot_smooth` | off, on |

## Criterion

- **Primary:** the worst case over the two sets of the mean frame accuracy (training subset,
  p = 0.5).
- **Ties** go to the higher mean over the sets, then to the simpler setting: shot_smooth off, the
  smaller total weight, temperature 0.01.

## Reported

- **Untuned zero shot row:** equal weights 1, 1, 1, temperature 0.01, shot_smooth off. It is
  never chosen, only reported.
- **Each source alone:** the best siglip only, crop only and vlm only rows.
- **The test subset, once:** Engine A, B v1, the untuned row and the selected setting, with
  p = 0.5 and without cuts.

## QiSens

Khurshid runs siglip_scores and vlm_scores on the x run, then one compare:
- the pre chosen setting first;
- then the untuned equal weight row;
- B v1 as the reference.

## Disclosure added before any grid (2026-11-12)

A smoke run of vlm_scores on one Kallax video gave the first phase in every window (argmax agrees
with the GT on 22.2 % of its 18 windows). The prompt was not changed after it. The grid decides
the VLM weight, including 0.
