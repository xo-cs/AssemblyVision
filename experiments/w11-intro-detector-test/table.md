# w11-intro-detector-test

Intro detector on HA4M with still and sway intros (test split; rule w11-intro-detector)

Setting = intro_k, intro_sustain_s, intro_active_share. Gate: plain within 0.5 of off.

| setting | all | plain | still | sway | gate |
| --- | --- | --- | --- | --- | --- |
| 1,20,0.5 | 85.49 | 87.61 | 84.88 | 83.97 | fail |
| off | 78.02 | 88.53 | 72.77 | 72.77 | ref |

Test split, scored once with the setting chosen on the train split (1,20,0.5); nothing is chosen here (gate: plain within 0.5 of off, for information).
