# w10-engine-b-embed: choice rule, written before training Engine B v1 and before any grid run

Committed on 2026-11-08 before w10-engine-b-v1 is trained, so the choice cannot follow the results.

Grid (public runs only, CPU compare mode, tools/engine_b_grid.py):
- model engine_b/w10-engine-b-v1.pt (multitask: boundary head plus the 12 HA4M action head)
- seg_a.embed_weight in 0, 0.2, 0.4, 0.6, 0.8, 1.0
- seg_a.embed_window_s in 10, 20, 30, 60, 90, 120
- clips: process_480p30 (run_20261107_062727_311d, phases of about 1 min) and
  shots_check_480p30 (run_20261024_091734_e388, phases of about 10 s), their team made GT files

Rule:
1. A setting is allowed only if its frame accuracy is not below the weight 0 row on either clip.
2. It must be stable: the neighbouring weights (one step up and down) and windows (one
   step each side) at that point must also be allowed.
3. Among the allowed, stable settings, take the highest mean frame accuracy of the two
   clips; ties go to the lower boundary MAE on process_480p30, then the smaller weight,
   then the longer window.
4. If no setting beats the weight 0 row on that mean, the choice is weight 0: Engine B v1
   is not used, and that is the recorded result.
The QiSens compare then runs the chosen setting as the headline; every other row is post
hoc sensitivity, never used for choosing.
