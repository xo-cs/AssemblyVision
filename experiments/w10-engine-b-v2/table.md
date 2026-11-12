# w10-engine-b-v2: Engine B v2 (selected by the CV grid w10-engine-b-v2-cv): contrast window 4 s, weight 1.0, on the 152 train videos

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation), cross subject split.

Main fold cross_subject: train 152 videos of 28 subjects, test 65 videos of 13 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cross_subject | all | 0.2 | 0.6877 | 0.7284 | 0.7074 | 1236 | 1167 |
| cross_subject | all | 0.5 | 0.8568 | 0.9075 | 0.8814 | 1236 | 1167 |
| cross_subject | lab_a | 0.2 | 0.692 | 0.7276 | 0.7094 | 552 | 525 |
| cross_subject | lab_a | 0.5 | 0.8587 | 0.9029 | 0.8802 | 552 | 525 |
| cross_subject | lab_b | 0.2 | 0.6842 | 0.729 | 0.7059 | 684 | 642 |
| cross_subject | lab_b | 0.5 | 0.8553 | 0.9112 | 0.8824 | 684 | 642 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.6969 | 0.8703 |
| 0.4 | 0.7033 | 0.8766 |
| 0.5 | 0.7074 | 0.8814 |
| 0.6 | 0.7121 | 0.8861 |
| 0.7 | 0.7166 | 0.8868 |

- cross_subject: model models/engine_b/w10-engine-b-v2.pt (sha256 d40625bfc2e5afc7...), 152 train / 65 test videos, pos_weight 9.914, 60 epochs, final train loss 4.77561, 101.6 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cross_subject 0.8998.

336410 parameters per model, trained on NVIDIA RTX A5000.
