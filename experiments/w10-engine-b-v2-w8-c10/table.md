# w10-engine-b-v2-w8-c10: Engine B v2 on the CV folds: contrast window 8 s, weight 1.0

First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints mapped to COCO 17, YOLO like augmentation), cross subject split.

Main fold cross_subject: train 97 videos of 18 subjects, test 55 videos of 10 held out subjects (ha4m_subject_split.json); cross location folds from ha4m_splits.json (train one lab, test the other). 10 fps; targets Gaussian sigma 0.2 s; peaks at 0.5, 0.5 s apart; hip calibration off.

| fold | test set | tolerance s | precision | recall | F1 | predicted | true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cv1 | all | 0.2 | 0.6422 | 0.7035 | 0.6715 | 1090 | 995 |
| cv1 | all | 0.5 | 0.8147 | 0.8925 | 0.8518 | 1090 | 995 |
| cv1 | lab_a | 0.2 | 0.6469 | 0.6921 | 0.6688 | 490 | 458 |
| cv1 | lab_a | 0.5 | 0.849 | 0.9083 | 0.8776 | 490 | 458 |
| cv1 | lab_b | 0.2 | 0.6383 | 0.7132 | 0.6737 | 600 | 537 |
| cv1 | lab_b | 0.5 | 0.7867 | 0.879 | 0.8303 | 600 | 537 |
| cv2 | all | 0.2 | 0.6259 | 0.6598 | 0.6424 | 1029 | 976 |
| cv2 | all | 0.5 | 0.8358 | 0.8811 | 0.8579 | 1029 | 976 |
| cv2 | lab_a | 0.2 | 0.6181 | 0.6356 | 0.6267 | 618 | 601 |
| cv2 | lab_a | 0.5 | 0.8495 | 0.8735 | 0.8614 | 618 | 601 |
| cv2 | lab_b | 0.2 | 0.6375 | 0.6987 | 0.6667 | 411 | 375 |
| cv2 | lab_b | 0.5 | 0.8151 | 0.8933 | 0.8524 | 411 | 375 |
| cv3 | all | 0.2 | 0.6448 | 0.7152 | 0.6782 | 853 | 769 |
| cv3 | all | 0.5 | 0.8335 | 0.9246 | 0.8767 | 853 | 769 |
| cv3 | lab_a | 0.2 | 0.6332 | 0.6997 | 0.6648 | 379 | 343 |
| cv3 | lab_a | 0.5 | 0.8364 | 0.9242 | 0.8781 | 379 | 343 |
| cv3 | lab_b | 0.2 | 0.654 | 0.7277 | 0.6889 | 474 | 426 |
| cv3 | lab_b | 0.5 | 0.8312 | 0.9249 | 0.8756 | 474 | 426 |

F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):

| threshold | F1@0.2 | F1@0.5 |
| --- | --- | --- |
| 0.3 | 0.6589 | 0.8358 |
| 0.4 | 0.667 | 0.8454 |
| 0.5 | 0.6715 | 0.8518 |
| 0.6 | 0.677 | 0.8594 |
| 0.7 | 0.6841 | 0.8674 |

- cv1: model models/engine_b/w10-engine-b-v2-w8-c10.cv1.pt (sha256 e0e466f6dbdde532...), 97 train / 55 test videos, pos_weight 9.697, 60 epochs, final train loss 3.86976, 45.4 s.
- cv2: model models/engine_b/w10-engine-b-v2-w8-c10.cv2.pt (sha256 f6136d0c02937fb4...), 98 train / 54 test videos, pos_weight 10.082, 60 epochs, final train loss 3.94229, 45.3 s.
- cv3: model models/engine_b/w10-engine-b-v2-w8-c10.cv3.pt (sha256 b24a24bd795d87a6...), 109 train / 43 test videos, pos_weight 9.956, 60 epochs, final train loss 3.74333, 49.2 s.

Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test videos: cv1 0.8706, cv2 0.8877, cv3 0.8738.

336410 parameters per model, trained on NVIDIA RTX A5000.
