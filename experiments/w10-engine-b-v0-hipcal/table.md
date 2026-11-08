# w10-engine-b-v0-hipcal: Engine B v0 with hip calibration

As w10-engine-b-v0, with engine_b.hip_calibration on: every Kinect joint moved by the mean YOLO26 x offset of that joint over both labs (w10-ha4m-domain-gap, body frame, torso lengths) before the features, in training and test data.

Main fold cross_subject: train 152 videos of 28 subjects, test 65 videos of 13 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration on (w10-ha4m-domain-gap, mean of both labs per joint).

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross_subject | all | 0.2 | 0.7291 | 0.7609 | 0.7447 | 1218 | 1167 |
| cross_subject | all | 0.5 | 0.8686 | 0.9066 | 0.8872 | 1218 | 1167 |
| cross_subject | lab_a | 0.2 | 0.718 | 0.7371 | 0.7274 | 539 | 525 |
| cross_subject | lab_a | 0.5 | 0.8701 | 0.8933 | 0.8816 | 539 | 525 |
| cross_subject | lab_b | 0.2 | 0.7378 | 0.7804 | 0.7585 | 679 | 642 |
| cross_subject | lab_b | 0.5 | 0.8675 | 0.9174 | 0.8917 | 679 | 642 |
| cross_location_test_lab_a | all | 0.2 | 0.6253 | 0.6373 | 0.6312 | 1964 | 1927 |
| cross_location_test_lab_a | all | 0.5 | 0.776 | 0.7909 | 0.7833 | 1964 | 1927 |
| cross_location_test_lab_a | lab_a | 0.2 | 0.6253 | 0.6373 | 0.6312 | 1964 | 1927 |
| cross_location_test_lab_a | lab_a | 0.5 | 0.776 | 0.7909 | 0.7833 | 1964 | 1927 |
| cross_location_test_lab_b | all | 0.2 | 0.5165 | 0.5545 | 0.5348 | 2126 | 1980 |
| cross_location_test_lab_b | all | 0.5 | 0.7248 | 0.7783 | 0.7506 | 2126 | 1980 |
| cross_location_test_lab_b | lab_b | 0.2 | 0.5165 | 0.5545 | 0.5348 | 2126 | 1980 |
| cross_location_test_lab_b | lab_b | 0.5 | 0.7248 | 0.7783 | 0.7506 | 2126 | 1980 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.7406 | 0.8853 |
| 0.4 | 0.7426 | 0.8869 |
| 0.5 | 0.7447 | 0.8872 |
| 0.6 | 0.7475 | 0.891 |
| 0.7 | 0.7507 | 0.8934 |

- cross_subject: model models/engine_b/w10-engine-b-v0-hipcal.pt (sha256 22195e5b085d3483...), 152 train / 65 test videos, pos_weight 9.914, 60 epochs, final train loss 0.81811, 80.0 s.
- cross_location_test_lab_a: model models/engine_b/w10-engine-b-v0-hipcal.cross_location_test_lab_a.pt (sha256 d9dc8468800a375c...), 111 train / 106 test videos, pos_weight 7.54, 60 epochs, final train loss 0.93903, 51.8 s.
- cross_location_test_lab_b: model models/engine_b/w10-engine-b-v0-hipcal.cross_location_test_lab_b.pt (sha256 000623cfc64a880b...), 106 train / 111 test videos, pos_weight 12.419, 60 epochs, final train loss 0.72379, 63.2 s.

334082 parameters per model, trained on NVIDIA RTX A5000.
