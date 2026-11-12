# w10-engine-b-v2-w4-c05: Engine B v2 on the CV folds: contrast window 4 s, weight 0.5

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation), cross subject split.

Main fold cross_subject: train 97 videos of 18 subjects, test 55 videos of 10 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv1 | all | 0.2 | 0.6421 | 0.7015 | 0.6705 | 1087 | 995 |
| cv1 | all | 0.5 | 0.8224 | 0.8985 | 0.8588 | 1087 | 995 |
| cv1 | lab_a | 0.2 | 0.6551 | 0.7009 | 0.6772 | 490 | 458 |
| cv1 | lab_a | 0.5 | 0.8469 | 0.9061 | 0.8755 | 490 | 458 |
| cv1 | lab_b | 0.2 | 0.6315 | 0.702 | 0.6649 | 597 | 537 |
| cv1 | lab_b | 0.5 | 0.8023 | 0.892 | 0.8448 | 597 | 537 |
| cv2 | all | 0.2 | 0.638 | 0.6609 | 0.6492 | 1011 | 976 |
| cv2 | all | 0.5 | 0.8576 | 0.8883 | 0.8727 | 1011 | 976 |
| cv2 | lab_a | 0.2 | 0.6393 | 0.6489 | 0.6441 | 610 | 601 |
| cv2 | lab_a | 0.5 | 0.8721 | 0.8852 | 0.8786 | 610 | 601 |
| cv2 | lab_b | 0.2 | 0.6359 | 0.68 | 0.6572 | 401 | 375 |
| cv2 | lab_b | 0.5 | 0.8354 | 0.8933 | 0.8634 | 401 | 375 |
| cv3 | all | 0.2 | 0.6614 | 0.7035 | 0.6818 | 818 | 769 |
| cv3 | all | 0.5 | 0.857 | 0.9116 | 0.8834 | 818 | 769 |
| cv3 | lab_a | 0.2 | 0.6521 | 0.6939 | 0.6723 | 365 | 343 |
| cv3 | lab_a | 0.5 | 0.8521 | 0.9067 | 0.8785 | 365 | 343 |
| cv3 | lab_b | 0.2 | 0.6689 | 0.7113 | 0.6894 | 453 | 426 |
| cv3 | lab_b | 0.5 | 0.8609 | 0.9155 | 0.8874 | 453 | 426 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.6573 | 0.8427 |
| 0.4 | 0.6657 | 0.8519 |
| 0.5 | 0.6705 | 0.8588 |
| 0.6 | 0.6764 | 0.8649 |
| 0.7 | 0.6817 | 0.8684 |

- cv1: model models/engine_b/w10-engine-b-v2-w4-c05.cv1.pt (sha256 02ba430290a5df5b...), 97 train / 55 test videos, pos_weight 9.697, 60 epochs, final train loss 3.21247, 45.5 s.
- cv2: model models/engine_b/w10-engine-b-v2-w4-c05.cv2.pt (sha256 776f60fc4f1199b4...), 98 train / 54 test videos, pos_weight 10.082, 60 epochs, final train loss 3.20932, 44.9 s.
- cv3: model models/engine_b/w10-engine-b-v2-w4-c05.cv3.pt (sha256 deeb8ccd3ec2ab9a...), 109 train / 43 test videos, pos_weight 9.956, 60 epochs, final train loss 3.06041, 48.8 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cv1 0.8698, cv2 0.8898, cv3 0.8772.

336410 parameters per model, trained on NVIDIA RTX A5000.
