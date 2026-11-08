# w10-engine-b-v1: Engine B v1: multitask MS TCN (boundary plus 12 actions), HA4M

As w10-engine-b-v0 with the 12 action head, engine_b.action_weight 1.0.

Main fold cross_subject: train 152 videos of 28 subjects, test 65 videos of 13 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross_subject | all | 0.2 | 0.7164 | 0.7446 | 0.7303 | 1213 | 1167 |
| cross_subject | all | 0.5 | 0.8788 | 0.9135 | 0.8958 | 1213 | 1167 |
| cross_subject | lab_a | 0.2 | 0.7276 | 0.7581 | 0.7425 | 547 | 525 |
| cross_subject | lab_a | 0.5 | 0.8647 | 0.901 | 0.8825 | 547 | 525 |
| cross_subject | lab_b | 0.2 | 0.7072 | 0.7336 | 0.7202 | 666 | 642 |
| cross_subject | lab_b | 0.5 | 0.8904 | 0.9237 | 0.9067 | 666 | 642 |
| cross_location_test_lab_a | all | 0.2 | 0.6043 | 0.6087 | 0.6065 | 1941 | 1927 |
| cross_location_test_lab_a | all | 0.5 | 0.7625 | 0.768 | 0.7653 | 1941 | 1927 |
| cross_location_test_lab_a | lab_a | 0.2 | 0.6043 | 0.6087 | 0.6065 | 1941 | 1927 |
| cross_location_test_lab_a | lab_a | 0.5 | 0.7625 | 0.768 | 0.7653 | 1941 | 1927 |
| cross_location_test_lab_b | all | 0.2 | 0.502 | 0.5091 | 0.5055 | 2008 | 1980 |
| cross_location_test_lab_b | all | 0.5 | 0.746 | 0.7566 | 0.7513 | 2008 | 1980 |
| cross_location_test_lab_b | lab_b | 0.2 | 0.502 | 0.5091 | 0.5055 | 2008 | 1980 |
| cross_location_test_lab_b | lab_b | 0.5 | 0.746 | 0.7566 | 0.7513 | 2008 | 1980 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.7219 | 0.8885 |
| 0.4 | 0.7271 | 0.8926 |
| 0.5 | 0.7303 | 0.8958 |
| 0.6 | 0.7335 | 0.8993 |
| 0.7 | 0.7361 | 0.8975 |

- cross_subject: model models/engine_b/w10-engine-b-v1.pt (sha256 9e8f18f2103bd17b...), 152 train / 65 test videos, pos_weight 9.914, 60 epochs, final train loss 1.29515, 58.1 s.
- cross_location_test_lab_a: model models/engine_b/w10-engine-b-v1.cross_location_test_lab_a.pt (sha256 c33a370666999e8d...), 111 train / 106 test videos, pos_weight 7.54, 60 epochs, final train loss 1.47421, 37.2 s.
- cross_location_test_lab_b: model models/engine_b/w10-engine-b-v1.cross_location_test_lab_b.pt (sha256 7f10782cc252ef54...), 106 train / 111 test videos, pos_weight 12.419, 60 epochs, final train loss 1.16325, 46.1 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cross_subject 0.9084, cross_location_test_lab_a 0.767, cross_location_test_lab_b 0.6361.

336410 parameters per model, trained on NVIDIA RTX A5000.
