# w10-engine-b-v2-w8-c05: Engine B v2 on the CV folds: contrast window 8 s, weight 0.5

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation), cross subject split.

Main fold cross_subject: train 97 videos of 18 subjects, test 55 videos of 10 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv1 | all | 0.2 | 0.6391 | 0.6925 | 0.6647 | 1078 | 995 |
| cv1 | all | 0.5 | 0.8247 | 0.8935 | 0.8577 | 1078 | 995 |
| cv1 | lab_a | 0.2 | 0.6426 | 0.679 | 0.6603 | 484 | 458 |
| cv1 | lab_a | 0.5 | 0.8595 | 0.9083 | 0.8832 | 484 | 458 |
| cv1 | lab_b | 0.2 | 0.6364 | 0.7039 | 0.6684 | 594 | 537 |
| cv1 | lab_b | 0.5 | 0.7963 | 0.8808 | 0.8364 | 594 | 537 |
| cv2 | all | 0.2 | 0.6571 | 0.6793 | 0.668 | 1009 | 976 |
| cv2 | all | 0.5 | 0.8563 | 0.8852 | 0.8705 | 1009 | 976 |
| cv2 | lab_a | 0.2 | 0.66 | 0.6622 | 0.6611 | 603 | 601 |
| cv2 | lab_a | 0.5 | 0.8706 | 0.8735 | 0.8721 | 603 | 601 |
| cv2 | lab_b | 0.2 | 0.6527 | 0.7067 | 0.6786 | 406 | 375 |
| cv2 | lab_b | 0.5 | 0.835 | 0.904 | 0.8681 | 406 | 375 |
| cv3 | all | 0.2 | 0.6539 | 0.71 | 0.6808 | 835 | 769 |
| cv3 | all | 0.5 | 0.8431 | 0.9155 | 0.8778 | 835 | 769 |
| cv3 | lab_a | 0.2 | 0.6368 | 0.7055 | 0.6694 | 380 | 343 |
| cv3 | lab_a | 0.5 | 0.8368 | 0.9271 | 0.8797 | 380 | 343 |
| cv3 | lab_b | 0.2 | 0.6681 | 0.7136 | 0.6901 | 455 | 426 |
| cv3 | lab_b | 0.5 | 0.8484 | 0.9061 | 0.8763 | 455 | 426 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.6522 | 0.8433 |
| 0.4 | 0.6594 | 0.8516 |
| 0.5 | 0.6647 | 0.8577 |
| 0.6 | 0.6686 | 0.861 |
| 0.7 | 0.6745 | 0.8673 |

- cv1: model models/engine_b/w10-engine-b-v2-w8-c05.cv1.pt (sha256 dd1fca2166c9bed5...), 97 train / 55 test videos, pos_weight 9.697, 60 epochs, final train loss 2.66364, 46.4 s.
- cv2: model models/engine_b/w10-engine-b-v2-w8-c05.cv2.pt (sha256 c11fcb62de9deb70...), 98 train / 54 test videos, pos_weight 10.082, 60 epochs, final train loss 2.69292, 45.4 s.
- cv3: model models/engine_b/w10-engine-b-v2-w8-c05.cv3.pt (sha256 691d22e368e70944...), 109 train / 43 test videos, pos_weight 9.956, 60 epochs, final train loss 2.53527, 48.5 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cv1 0.8692, cv2 0.8883, cv3 0.8803.

336410 parameters per model, trained on NVIDIA RTX A5000.
