# w10-engine-b-v1-cv: Engine B v1 recipe on the 3 CV folds of the train split (v2 rule)

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation), cross subject split.

Main fold cross_subject: train 97 videos of 18 subjects, test 55 videos of 10 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv1 | all | 0.2 | 0.6414 | 0.6975 | 0.6683 | 1082 | 995 |
| cv1 | all | 0.5 | 0.8272 | 0.8995 | 0.8618 | 1082 | 995 |
| cv1 | lab_a | 0.2 | 0.6455 | 0.6878 | 0.666 | 488 | 458 |
| cv1 | lab_a | 0.5 | 0.8545 | 0.9105 | 0.8816 | 488 | 458 |
| cv1 | lab_b | 0.2 | 0.638 | 0.7058 | 0.6702 | 594 | 537 |
| cv1 | lab_b | 0.5 | 0.8047 | 0.8901 | 0.8453 | 594 | 537 |
| cv2 | all | 0.2 | 0.6587 | 0.6803 | 0.6694 | 1008 | 976 |
| cv2 | all | 0.5 | 0.8641 | 0.8924 | 0.878 | 1008 | 976 |
| cv2 | lab_a | 0.2 | 0.6584 | 0.6606 | 0.6595 | 603 | 601 |
| cv2 | lab_a | 0.5 | 0.8789 | 0.8819 | 0.8804 | 603 | 601 |
| cv2 | lab_b | 0.2 | 0.6593 | 0.712 | 0.6846 | 405 | 375 |
| cv2 | lab_b | 0.5 | 0.842 | 0.9093 | 0.8744 | 405 | 375 |
| cv3 | all | 0.2 | 0.6877 | 0.7217 | 0.7043 | 807 | 769 |
| cv3 | all | 0.5 | 0.8625 | 0.9051 | 0.8832 | 807 | 769 |
| cv3 | lab_a | 0.2 | 0.6759 | 0.7114 | 0.6932 | 361 | 343 |
| cv3 | lab_a | 0.5 | 0.8643 | 0.9096 | 0.8864 | 361 | 343 |
| cv3 | lab_b | 0.2 | 0.6973 | 0.73 | 0.7133 | 446 | 426 |
| cv3 | lab_b | 0.5 | 0.861 | 0.9014 | 0.8807 | 446 | 426 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.6576 | 0.8497 |
| 0.4 | 0.6638 | 0.8565 |
| 0.5 | 0.6683 | 0.8618 |
| 0.6 | 0.6738 | 0.8676 |
| 0.7 | 0.6784 | 0.8719 |

- cv1: model models/engine_b/w10-engine-b-v1-cv.cv1.pt (sha256 99a71fdad2a29199...), 97 train / 55 test videos, pos_weight 9.697, 60 epochs, final train loss 1.47412, 45.1 s.
- cv2: model models/engine_b/w10-engine-b-v1-cv.cv2.pt (sha256 43a08d30e5e21f15...), 98 train / 54 test videos, pos_weight 10.082, 60 epochs, final train loss 1.44622, 43.5 s.
- cv3: model models/engine_b/w10-engine-b-v1-cv.cv3.pt (sha256 e112aafe07ca3dc5...), 109 train / 43 test videos, pos_weight 9.956, 60 epochs, final train loss 1.33107, 49.0 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cv1 0.873, cv2 0.8924, cv3 0.883.

336410 parameters per model, trained on NVIDIA RTX A5000.
