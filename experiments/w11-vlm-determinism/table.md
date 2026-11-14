# w11-vlm-determinism

VLM repeat determinism (Qwen3 VL 8B Q4_K_M, llama.cpp b12005, GPU1). Greedy decoding (temperature 0), fixed seed 42
on the server and in every request. Two IKEA ASM videos (Kallax 0002 white floor, Lack Coffee 0001 oak table; 70
windows of 3 s with 2 frames), two repeats per mode, each repeat a fresh llama-server.

| mode | windows identical | replies identical | max probability difference | argmax agrees | s per window |
| --- | --- | --- | --- | --- | --- |
| 4 slots, prompt cache on (default) | 66/70 | 61/70 | 0.010 | yes | 0.78 to 0.80 |
| 4 slots, prompt cache off | 63/70 | 59/70 | 0.010 | yes | 0.80 to 0.83 |
| 1 slot, prompt cache off | 70/70 | 70/70 | 0.000 | yes | 1.39 |

Findings:

- With one slot the outputs are bit identical between repeats. With parallel slots they are not: requests are
  batched together in different combinations, which changes the floating point sums, so a few windows (4 to 7 of
  70) differ by up to 0.01 in a probability; the most likely phase did not change in these repeats.
- Turning the prompt cache off does not remove the differences, so it stays on.
- One slot is 1.7x slower (1.39 against 0.80 s per window): on QiSens about 275 s of VLM instead of about 160 s,
  longer than the pose pass, so the full run would get longer.
- Default kept: 4 slots (speed). For a bit identical rerun: --set zero_shot.vlm_slots=1.
