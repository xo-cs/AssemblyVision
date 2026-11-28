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

## Amendment of group c (2026-11-28, before any generation)

Khurshid's decision: the human authors are replaced by **three local text only language models of different families**,
because no independent human authors are available. Fixed here, before any model writes anything:

- **Authors:** Qwen3 4B, Llama 3.2 3B Instruct and Gemma 3 4B it, GGUF Q4_K_M, run with our llama.cpp (llama-server) on
  GPU1. If one is unavailable, another small model of a different family replaces it and the reason is recorded.
- **Input:** only the prompt below, the same text for all three, as one user message through each model's own chat
  template. No image, video, result, run or other file goes into any model. Qwen3's thinking mode is switched off through
  the chat template (enable_thinking = false), so the prompt stays identical.
- **Sampling:** temperature 0.7, seed 20261128, at most 4,096 new tokens, **one generation per model**. No second try.
- **Output:** saved raw (experiments/w12-robustness/group_c/), then only format fixes (the exact phase headings, the
  "- " line prefix, removing text outside the five sections); no sentence is added, removed or reworded. If a model writes
  more or fewer than 22 lines for a phase, they are kept as written and the count is recorded.
- **Scoring:** the frozen method; only the SigLIP text changes, exactly as groups b and c are defined above (the VLM prompt
  keeps the frozen short descriptions). Verdict rule unchanged.
- **Limit:** language models may share knowledge with the model that wrote the frozen descriptions, so this tests the
  wording, not fully independent authors.

The prompt:

```
You write visual descriptions for a video classifier. Below are 5 work phases of train bogie maintenance in a factory. For each phase, write 22 short sentences describing what a camera would SEE during that phase: tools, parts, what the hands and body do, the state of the parts, wide and close views. Do not use the phase names inside the sentences.
1. Removal: the worker takes components off the train or bogie (for example wheelsets, covers, brake parts) and moves them away.
2. Disassembly: the worker takes the removed components apart into smaller parts (unbolting, separating, opening housings).
3. Cleaning: the worker cleans the parts (washing, brushing, wiping, blowing off dirt, grease or rust).
4. Adjustment and Inspection: the worker checks and measures the parts (gauges, measuring tools, visual checks) and adjusts them.
5. Assembly and Installation: the worker puts the parts back together and mounts them onto the bogie or train (fitting, bolting, tightening).
Format exactly: '## <phase name>' then 22 lines starting with '- '.
```

### Second amendment of group c (2026-11-28, before the replacement is generated)

Qwen3 4B stopped after one phase (incomplete). Replaced by Phi 4 mini instruct (different family), decided by Khurshid on
2026-11-28 after seeing only Qwen3's incomplete output, before any group c score existed. Qwen3's output is kept in
group_c/ as a record. Same prompt, temperature, seed and one generation rule; Phi 4 mini instruct as GGUF Q4_K_M with our
llama.cpp on GPU1. If its output is incomplete too, it is not rerun: group c is then scored on the complete authors and
the report says so.
