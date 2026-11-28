# w12-robustness

Robustness study of the flagship (rule experiments/w12-robustness/rule.md, written before any variant ran), run by
Khurshid on QiSens, approved summary numbers as he reported them (2026-11-27). Freeze validation-freeze-1, method
unchanged: True. Reference 93.20; a variant is within tolerance at 88.20 to 98.20.

| group | n | min | median | max | within 5 points | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| a. weights (vlm, siglip, crop at 0.5x and 2x; temperature at 0.5x and 2x) | 8 | 90.04 | 93.20 | 94.40 | 100 % | robust |
| b. description subsets (11 of 22 per phase, 20 seeds) | 20 | 90.55 | 93.20 | 94.40 | 100 % | robust |
| c. independent descriptions (Asel, Kazi, Huzaifa) | - | - | - | - | - | **not tested yet** |
| d. VLM sampling (3 s windows, 2 frames) | 1 | 93.40 | 93.40 | 93.40 | 100 % | robust |
| e. pose model (YOLO26 l) | 1 | 94.01 | 94.01 | 94.01 | 100 % | robust |

**Overall: robust in every tested group; group c is pending**, so "robust overall" is not claimed yet (rule).

Limits (from the rule): this is stability on the one video, not an estimate for new videos. Groups b and c change only
the SigLIP text, not the VLM prompt, and the VLM carries most of the gain.

## Group c attempts (2026-11-28)

Authors are local text only LLMs (rule.md, amendments of 2026-11-28, each committed before its generation); one
generation each, same prompt, temperature 0.7, seed 20261128, at most 4,096 new tokens. Raw outputs and commands:
group_c/.

| order | model | family | lines per phase (Removal, Disassembly, Cleaning, Adjustment and Inspection, Assembly and Installation) | complete | used |
| --- | --- | --- | --- | --- | --- |
| 1 | Qwen3 4B | Qwen (Alibaba) | 22, -, -, -, - (stopped after the first phase) | no | no, replaced |
| 2 | Llama 3.2 3B Instruct | Llama (Meta) | 23, 23, 23, 23, 23 | yes | yes |
| 3 | Gemma 3 4B it | Gemma (Google) | 20, 17, 16, 15, 15 | yes | yes |
| 4 | Phi 4 mini instruct | Phi (Microsoft) | 36, 28, 38, 27, 275 (repetition: 36, 25, 19, 12, 7 distinct; cut at 4,096 tokens) | no | no, replaced |
| 5 | Mistral 7B Instruct v0.3 | Mistral | 21, 21, 21, 21, 21 | yes | yes |

Group c is scored on Llama 3.2, Gemma 3 and Mistral 7B. The two replacements were decided by Khurshid after seeing only the
incomplete outputs, before any group c score existed.
