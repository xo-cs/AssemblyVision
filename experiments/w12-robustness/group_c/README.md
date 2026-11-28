# Group c: phase descriptions by three local text LLMs

Rule: experiments/w12-robustness/rule.md (amendment of 2026-11-28, committed before any generation). Generated on
2026-11-28 with generate.py: llama-server (third_party/llama.cpp) on GPU1, one request per model, temperature 0.7, seed
20261128, at most 4,096 new tokens, the same prompt for all three, Qwen3's thinking mode off through the chat template.
No image, video or file of a run went into any model.

| key | model | file (models/text_llm) | SHA256 | licence | output |
| --- | --- | --- | --- | --- | --- |
| llm_qwen3 | Qwen3 4B, Q4_K_M (Qwen/Qwen3-4B-GGUF) | Qwen3-4B-Q4_K_M.gguf | 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 | Apache 2.0 | **incomplete:** only the first phase (Removal, 22 lines), then the model stopped (finish reason "stop", 375 tokens). Not scored; no second try (rule). |
| llm_llama32 | Llama 3.2 3B Instruct, Q4_K_M (bartowski/Llama-3.2-3B-Instruct-GGUF, a conversion of Meta's weights) | Llama-3.2-3B-Instruct-Q4_K_M.gguf | 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff | Llama 3.2 Community License | all 5 phases, 23 lines each (asked for 22; kept as written) |
| llm_gemma3 | Gemma 3 4B it, Q4_K_M (bartowski/google_gemma-3-4b-it-GGUF, a conversion of Google's weights) | google_gemma-3-4b-it-Q4_K_M.gguf | 4996030242583a40aa151ff93f49ed787ac8c25e4120c3ae4588b2e2a7d1ae94 | Gemma Terms of Use | all 5 phases, 20, 17, 16, 15 and 15 lines (fewer than 22; kept as written) |

Files: <key>.cmd.txt (the exact server command and request), <key>.raw.json (the response), <key>.raw.md (the text as
written), <key>.server.log. The format fixed versions are docs_team/independent_descriptions/<key>.md (headings and the
"- " prefix only; a check confirmed that the sentences are unchanged). The checksums match the Hugging Face listings.

So group c is scored on two authors (Llama 3.2 and Gemma 3); the robustness tool lists Qwen3 as incomplete, not scored.

## Replacement of Qwen3 (second amendment of the rule, 2026-11-28)

| key | model | file (models/text_llm) | SHA256 | licence | output |
| --- | --- | --- | --- | --- | --- |
| llm_phi4mini | Phi 4 mini instruct, Q4_K_M (bartowski/microsoft_Phi-4-mini-instruct-GGUF; Microsoft publishes no GGUF) | microsoft_Phi-4-mini-instruct-Q4_K_M.gguf | 01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2 | MIT | **incomplete:** all five headings, but the model fell into repetition and was cut off at the 4,096 token limit (finish reason "length", the last line breaks off mid sentence). Lines per phase, distinct in brackets: Removal 36 (36), Disassembly 28 (25), Cleaning 38 (19), Adjustment and Inspection 27 (12), Assembly and Installation 275 (7). |

Following Khurshid's instruction, it was not rerun and not format fixed, and it is not used for group c. Removing the
repeated lines would remove sentences, which the rule does not allow. Group c stays scored on Llama 3.2 and Gemma 3.

## Third author (third amendment of the rule, 2026-11-28)

| key | model | file (models/text_llm) | SHA256 | licence | output |
| --- | --- | --- | --- | --- | --- |
| llm_mistral7b | Mistral 7B Instruct v0.3, Q4_K_M (bartowski/Mistral-7B-Instruct-v0.3-GGUF; Mistral publishes no GGUF) | Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | 1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6 | Apache 2.0 | complete: all five phases, 21 distinct lines each (asked for 22; kept as written), finish reason "stop", 2,912 tokens |

Format fix: the model numbered its lines (1. to 21.); the numbers were replaced by the "- " prefix. Group c is now scored on
Llama 3.2, Gemma 3 and Mistral 7B.
