# VLM speed selection rule (written 2026-11-13, before any setting of the grid is scored)

**Goal.** A faster vlm_scores stage for the default step engine "B zero shot", accuracy first. QiSens
is never used to choose (QiSens blind result so far: 93.20 with the current setting, w11-qisens-zero-shot).

## Current setting (the reference)

- Window 2 s, 4 frames per window, 448 px, 2 llama-server slots (the vlm.npz files of rule w11-zero-shot).
- Prompt as frozen, only the span follows the window ("about two seconds", "about four seconds"). Disclosed:
  the first line of the prompt names the number of phases, not of frames ("These 5 frames" on QiSens with
  4 frames sent). It is kept as it is, so the current setting stays the one measured on QiSens.

## Step 1: slots (speed only)

- llama-server slots (parallel requests) 1, 2 and 4 on the same 3 IKEA training videos, the current
  window and frames.
- Chosen: the highest throughput (windows per second). Ties within 5 % go to fewer slots.
- Every grid setting of step 2 then runs with the chosen slots, the current setting included, so a
  change of the outputs by batching is measured as part of the grid.

## Step 2: grid (window and frames)

| setting | values |
| --- | --- |
| `zero_shot.vlm_window_s` (one request per window, no overlap) | 2, 3, 4 |
| `zero_shot.vlm_frames` (evenly over the window) | 4, 2 |

**Videos.** 20 Kallax Shelf Drawer and 20 Lack Coffee Table videos of the official training subset with
siglip.npz and vlm.npz, drawn with seed 42 (stored in the result). The testing subset is not used.

**Accuracy.** Mean frame accuracy per set with the untuned equal weights of "B zero shot" (siglip 1,
crop 1, vlm 1, temperature 0.01, shot_smooth off, B v1 boundary evidence, dense candidates), the
synthetic cuts p 0.5 (as in w11-zero-shot). The measure is the worst case over the two sets.

**Speed.** VLM scoring seconds per minute of video, summed over the 40 videos (the server start not
included), on GPU1 with nothing else on it.

## Criterion

- **Reference accuracy:** the stored vlm.npz files (current setting, 2 slots).
- **Chosen:** the fastest setting whose worst case accuracy is at most 1 point below the reference.
  The current setting with the chosen slots is a candidate too; if no setting passes, the current
  setting stays.
- **Reported:** every setting (accuracy per set, worst case, speed), and the expected QiSens full run
  time: the pose pass time of w10-qisens (and the timings Khurshid shared) against the VLM time from
  the measured speed at 9.9 minutes of video, plus SigLIP and the rest.

## QiSens

The chosen setting becomes the default of [zero_shot]. Khurshid runs one timing command of the full
default pipeline (3 repeats, with the GT), so its QiSens accuracy is measured by the eval of that run.
