# w10-start-of-work: start of work detection on the public clips

seg_a.start_detect (default off): phase 1 starts at the first sustained subject activity (at least 0.6 of the frames in 10 s moving, energy >= pause_energy), not at the first person frame; a still start shorter than start_min_intro_s (10 s) is not an intro; at most 0.25 of the active span.

Neither public GT file has an intro: process_480p30 GT starts at 3.0 s, exactly the first frame with the subject (0 to 3 s is empty, already "other"); shots_check_480p30 GT starts at 0. So the public clips can only test that the detector does no harm; the positive case is tested on synthetic timelines only (tests/test_engine_b_model.py, StartOfWorkTest).

| clip | setting | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| process_480p30 | start_detect = false | 92.33 | 100.00 | 100.00 | 100.00 | 100.00 | 2.50 |
| process_480p30 | start_detect = true, start_min_intro_s 0 (first version) | 90.39 | 100.00 | 100.00 | 100.00 | 100.00 | 3.29 |
| process_480p30 | start_detect = true | 92.33 | 100.00 | 100.00 | 100.00 | 100.00 | 2.50 |
| shots_check_480p30 | start_detect = false | 30.04 | 100.00 | 40.00 | 40.00 | 40.00 | 1.65 |
| shots_check_480p30 | start_detect = true, start_min_intro_s 0 (first version) | 27.40 | 100.00 | 40.00 | 40.00 | 20.00 | 1.65 |
| shots_check_480p30 | start_detect = true | 30.04 | 100.00 | 40.00 | 40.00 | 40.00 | 1.65 |

The first version cut false intros of 6.3 s (process) and 1.3 s (shots_check); start_min_intro_s = 10 s was added so that both clips stay unchanged. That value is set from clips without an intro (a no harm bound), not validated on a real intro.

Commands (CPU, in memory):

python -m assemblyvision.tools.run_pose --run ../data/public/runs/run_20261107_062727_311d --gt ../data/public/videos/process_480p30_gt.csv --compare seg_a.start_detect

python -m assemblyvision.tools.run_pose --run ../data/public/runs/run_20261024_091734_e388 --gt ../data/public/videos/shots_check_480p30_gt.csv --compare seg_a.start_detect
