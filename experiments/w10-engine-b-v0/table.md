# w10-engine-b-v0: Engine B v0: MS TCN boundary scorer, HA4M

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation); cross subject plus both cross location folds. Hip calibration off.

Main fold cross_subject: train 152 videos of 28 subjects, test 65 videos of 13 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross_subject | all | 0.2 | 0.7329 | 0.7618 | 0.7471 | 1213 | 1167 |
| cross_subject | all | 0.5 | 0.8747 | 0.9092 | 0.8916 | 1213 | 1167 |
| cross_subject | lab_a | 0.2 | 0.7278 | 0.7486 | 0.738 | 540 | 525 |
| cross_subject | lab_a | 0.5 | 0.8722 | 0.8971 | 0.8845 | 540 | 525 |
| cross_subject | lab_b | 0.2 | 0.737 | 0.7726 | 0.7544 | 673 | 642 |
| cross_subject | lab_b | 0.5 | 0.8767 | 0.919 | 0.8973 | 673 | 642 |
| cross_location_test_lab_a | all | 0.2 | 0.6193 | 0.6399 | 0.6294 | 1991 | 1927 |
| cross_location_test_lab_a | all | 0.5 | 0.775 | 0.8007 | 0.7876 | 1991 | 1927 |
| cross_location_test_lab_a | lab_a | 0.2 | 0.6193 | 0.6399 | 0.6294 | 1991 | 1927 |
| cross_location_test_lab_a | lab_a | 0.5 | 0.775 | 0.8007 | 0.7876 | 1991 | 1927 |
| cross_location_test_lab_b | all | 0.2 | 0.5279 | 0.5641 | 0.5454 | 2116 | 1980 |
| cross_location_test_lab_b | all | 0.5 | 0.7306 | 0.7808 | 0.7549 | 2116 | 1980 |
| cross_location_test_lab_b | lab_b | 0.2 | 0.5279 | 0.5641 | 0.5454 | 2116 | 1980 |
| cross_location_test_lab_b | lab_b | 0.5 | 0.7306 | 0.7808 | 0.7549 | 2116 | 1980 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.7419 | 0.8851 |
| 0.4 | 0.7451 | 0.8891 |
| 0.5 | 0.7471 | 0.8916 |
| 0.6 | 0.7465 | 0.8912 |
| 0.7 | 0.7493 | 0.894 |

- cross_subject: model models/engine_b/w10-engine-b-v0.pt (sha256 d8c7e3f073661f7e...), 152 train / 65 test videos, pos_weight 9.914, 60 epochs, final train loss 0.81171, 73.7 s.
- cross_location_test_lab_a: model models/engine_b/w10-engine-b-v0.cross_location_test_lab_a.pt (sha256 3a53fe837353e78f...), 111 train / 106 test videos, pos_weight 7.54, 60 epochs, final train loss 0.92376, 49.1 s.
- cross_location_test_lab_b: model models/engine_b/w10-engine-b-v0.cross_location_test_lab_b.pt (sha256 a01783b8f31633e9...), 106 train / 111 test videos, pos_weight 12.419, 60 epochs, final train loss 0.71932, 56.8 s.

334082 parameters per model, trained on NVIDIA RTX A5000.
