# Intro detector selection rule (written 2026-11-12, before any run of the detector on data)

**Detector.** `seg_a.intro_detect` (docs_team/intro_detector.md, rules.detect_intro). Motion
energy only, relative to the video's own working level. It is built general and validated on
public data only.

**Data.** The HA4M macro phase benchmark (experiments/datasets/ha4m_macro_phases.md):
- choice by CV on the 152 train videos;
- the 65 test videos once.

Every video is scored three times:
- plain;
- with a still intro (phase_bench.with_intro, mode still: the first pose held, 1 px jitter);
- with a sway intro (mode sway: the first pose held with a slow random drift of the whole body,
  steps of 0.5 px per frame, smoothed over 1 s; standing, no work).

The intro length is uniform from 5 to 30 s, seeded per video (seed 42). No outro is added: the
outro rule is the intro rule run backwards.

**Decoder.** The v2 selected decoder of w10-engine-b-v2-cv (CV models on train, the final v2 on
test), because there an intro costs most (88.53 to 72.77). Detector off is the reference.

**Grid.** `intro_k` 0.5, 1, 1.5 × `intro_sustain_s` 5, 10, 20 × `intro_active_share` 0.5, 0.6,
0.8. Fixed: `intro_smooth_s` 2, `intro_min_s` 3, `intro_max_share` 0.25.

**Criterion.** The mean frame accuracy over all 456 train video variants. Ties go to the larger
k (fewer false intros).
- **Gate:** the plain videos stay within 0.5 points of detector off. A setting outside the gate
  is not chosen. If none passes, the detector stays off.

**Also reported, once:**
- the test split with the chosen setting;
- the two public clips (B v1 default plus the detector; process_480p30 must stay 92.33).

**Default.** Off, until Khurshid decides.
