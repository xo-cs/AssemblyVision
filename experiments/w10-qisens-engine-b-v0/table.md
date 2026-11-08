# w10-qisens-engine-b-v0

QiSens video, Engine B v0 (engine_b/w10-engine-b-v0.pt) as evidence in Engine A, blind test: the setting was chosen on public runs before any QiSens result (w10-engine-b-pool). Approved summary numbers only (2026-11-08); everything else stays under the QiSens data area.

| setting | role | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | B peaks/min raw / pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weight 0 (Engine A alone) | baseline | 75.02 | 100 | 100 | 100 | 80 | 29.35 | - |
| **pool_s 30, weight 0.4** | pre chosen (w10-engine-b-pool, public runs only) | 55.51 | 100 | 60 | 60 | 20 | 27.13 | 4.95 / 0.51 |
| pool_s 0, weight 0.4 | post hoc sensitivity, never used for choosing | 79.96 | 100 | 100 | 100 | 60 | 23.45 | - |
| pool_s 10, weight 0.4 | post hoc sensitivity, never used for choosing | 65.52 | 100 | 80 | 80 | 60 | 27.13 | - |
| pool_s 60, weight 0.4 | post hoc sensitivity, never used for choosing | 49.72 | 100 | 60 | 40 | 20 | 37.24 | - |

Headline: the pre chosen setting lowers frame accuracy from 75.02 to 55.51 and F1@50 from 80 to 20; MAE drops a little (29.35 to 27.13 s). The pooled rate (0.51 peaks/min) was close to the planned phase rate, so the problem is where the peaks fall, not how often.

Conclusion: A fine action boundary scorer does not transfer to minute long phases.
