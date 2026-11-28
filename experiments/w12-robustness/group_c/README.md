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
