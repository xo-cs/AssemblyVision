# w10-engine-b-v2-w4-c10: Engine B v2 on the CV folds: contrast window 4 s, weight 1.0

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation), cross subject split.

Main fold cross_subject: train 97 videos of 18 subjects, test 55 videos of 10 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv1 | all | 0.2 | 0.6391 | 0.6905 | 0.6638 | 1075 | 995 |
| cv1 | all | 0.5 | 0.8307 | 0.8975 | 0.8628 | 1075 | 995 |
| cv1 | lab_a | 0.2 | 0.6473 | 0.6812 | 0.6638 | 482 | 458 |
| cv1 | lab_a | 0.5 | 0.8651 | 0.9105 | 0.8872 | 482 | 458 |
| cv1 | lab_b | 0.2 | 0.6324 | 0.6983 | 0.6637 | 593 | 537 |
| cv1 | lab_b | 0.5 | 0.8027 | 0.8864 | 0.8425 | 593 | 537 |
| cv2 | all | 0.2 | 0.6329 | 0.6588 | 0.6456 | 1016 | 976 |
| cv2 | all | 0.5 | 0.8445 | 0.8791 | 0.8614 | 1016 | 976 |
| cv2 | lab_a | 0.2 | 0.6295 | 0.6389 | 0.6342 | 610 | 601 |
| cv2 | lab_a | 0.5 | 0.8607 | 0.8735 | 0.8671 | 610 | 601 |
| cv2 | lab_b | 0.2 | 0.6379 | 0.6907 | 0.6633 | 406 | 375 |
| cv2 | lab_b | 0.5 | 0.8202 | 0.888 | 0.8528 | 406 | 375 |
| cv3 | all | 0.2 | 0.6416 | 0.6892 | 0.6646 | 826 | 769 |
| cv3 | all | 0.5 | 0.8499 | 0.9129 | 0.8803 | 826 | 769 |
| cv3 | lab_a | 0.2 | 0.624 | 0.6676 | 0.6451 | 367 | 343 |
| cv3 | lab_a | 0.5 | 0.8501 | 0.9096 | 0.8789 | 367 | 343 |
| cv3 | lab_b | 0.2 | 0.6558 | 0.7066 | 0.6802 | 459 | 426 |
| cv3 | lab_b | 0.5 | 0.8497 | 0.9155 | 0.8814 | 459 | 426 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.6482 | 0.8436 |
| 0.4 | 0.6578 | 0.8554 |
| 0.5 | 0.6638 | 0.8628 |
| 0.6 | 0.668 | 0.8701 |
| 0.7 | 0.6753 | 0.8759 |

- cv1: model models/engine_b/w10-engine-b-v2-w4-c10.cv1.pt (sha256 879d04cb84f2c9cb...), 97 train / 55 test videos, pos_weight 9.697, 60 epochs, final train loss 4.94227, 45.2 s.
- cv2: model models/engine_b/w10-engine-b-v2-w4-c10.cv2.pt (sha256 b136b6b16b67096f...), 98 train / 54 test videos, pos_weight 10.082, 60 epochs, final train loss 4.95993, 44.9 s.
- cv3: model models/engine_b/w10-engine-b-v2-w4-c10.cv3.pt (sha256 39ea3fb4ed834722...), 109 train / 43 test videos, pos_weight 9.956, 60 epochs, final train loss 4.82433, 49.7 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cv1 0.8702, cv2 0.8891, cv3 0.8756.

336410 parameters per model, trained on NVIDIA RTX A5000.
