# Robustness study rule (written 2026-11-21, before any variant is run)

**Question.** Does the 93.20 of the flagship depend on the exact settings and descriptions, or does it hold up
when they change? Asked by Khurshid after Professor Kim's review. There is no second QiSens video, so this says
how stable the result is **on this one video**. It is not an estimate for new videos.

**Code.** Git tag validation-freeze-1. Every result records the tag, the commit it ran on and whether the
method files are unchanged. The tool is tools/robustness.py. Khurshid runs it on QiSens; only approved summary
numbers enter this folder.

**Reference.** Flagship run_20261108_133251_1550, setting "B zero shot" (SigLIP full 1, crop 1, VLM 1, temperature
0.01), frame accuracy **93.20** against the final hand labels. Stored scores: SigLIP at 2 samples per second, VLM
windows of 2 s with 4 frames.

## Variants (fixed here, nothing is added or dropped after seeing a result)

| group | variants | n | runs on |
| --- | --- | --- | --- |
| a. weights | vlm, siglip and crop weight each at 0.5x and 2x (one changed at a time); siglip temperature at 0.5x and 2x | 8 | CPU |
| b. description subsets | for each variant and each phase, a random half of its descriptions (11 of 22; the phase file has 22 per phase, not 20); seeds 20261121 + i, i = 0 to 19 | 20 | CPU |
| c. independent descriptions | one file per author (Asel, Kazi, Huzaifa), each written from the phase definitions only, without seeing any result or docs_team/phase_descriptions.md | 3 | CPU |
| d. VLM sampling | 3 s windows with 2 frames against the stored 2 s with 4 frames | 1 | GPU1 for the scores (after the 4D redo), then CPU |
| e. pose model | the YOLO26 l run (run_20261025_053609_fd62) against x, with B zero shot: its SigLIP crops scored from its own tracking, the VLM scores of the same video | 1 | CPU |

**Limit of b and c.** Both change only the SigLIP text. The VLM prompt keeps the frozen short descriptions, because
new VLM scores need the GPU. So b and c test the SigLIP part only; the VLM, which carries most of the gain, is not
varied by them. This must be stated wherever b or c is quoted.

**Limit of c.** The authors are team members who know the project. It is a test of the wording, not of an outside party.

## Measure

Frame accuracy against the final hand labels (the evaluation rate of the headline), plus edit, F1@10/25/50 and MAE
per variant. Each group is reported as **n, minimum, median, maximum, and the share of variants within 5 points**.
Every variant is listed, not only the best.

## What counts as robust (fixed in advance)

- A variant is **within tolerance** if its frame accuracy is within 5 points of 93.20 (88.20 to 98.20).
- A group of several variants (a, b, c) is **robust** if at least 80 % of its variants are within tolerance.
- A single variant group (d, e) is **robust** if its variant is within tolerance.
- The result is called **robust overall** only if every group is robust. Otherwise the report names the factor
  that it is sensitive to. A group that was not run is reported as not tested, and "robust overall" is not claimed.
- The worst value of every group is reported next to the verdict, even when the group passes.

The rule is not changed after a result. A failed group is reported as it is, and 93.20 is then quoted together with that sensitivity.
