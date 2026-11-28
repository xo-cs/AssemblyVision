# COCO pose check of the trunk and wrist keypoints (rule written 2026-11-28, before the run)

**Question.** How well does our pose model find the keypoints behind measure 3 (trunk angle, waist bending) and measure 6
(hand travel, wrists)? Asked by Khurshid on 2026-11-28, so that these two measures are validated automatically on public
data instead of by hand clicks on QiSens frames.

**Data.** COCO 2017 val (5,000 images), person keypoints (annotations/person_keypoints_val2017.json), in
data/public/coco/. Public data only.

**Model and settings.** YOLO26 x pose (models/yolo26x-pose.pt, checksum in models/SHA256SUMS_yolo26_pose.txt), the
flagship's settings: image size 640, detector confidence 0.10, FP32 with TF32 (pose.half = false, pose.tf32 = true); one
image per call, because COCO images have different sizes (the flagship runs 8 frames of one video per call, which gives the
same result per frame). A predicted keypoint with confidence below 0.30 counts as missing (clean.body_conf_min, as in the
metrics). Code at tag validation-freeze-1 plus the new tool tools/coco_pose.py; the method files are unchanged.

**Persons.** Non crowd persons with all four of left and right shoulder and hip labelled visible (v = 2). A prediction is
matched to a person by box IoU of at least 0.5, greedily by detection confidence. A person without a match counts as a
miss: in PCK every joint of it is wrong, and it has no system trunk angle.

**Measures.**
- (a) PCK@0.1 per joint for left and right shoulder, hip and wrist: a joint labelled visible (v = 2) is correct if the
  predicted joint lies within 0.1 of the longer side of the person's COCO box. Over all persons (misses count) and over the
  matched persons only.
- (b) Trunk angle with analysis/motion.trunk_angle (mid shoulder minus mid hip against the upward image vertical) on the
  human keypoints and on the predicted ones: the mean absolute difference in degrees, exact band and within one band
  agreement (bands B0 under 20, B1 20 to 45, B2 45 to 60, B3 over 60), over the matched persons whose four predicted points
  are all found; the same for the bent subset (human trunk angle over 45 degrees). Coverage is reported.

**Targets** (frozen on 2026-11-26, not changed): #3 exact band at least 80 %, within one band at least 95 %; #6 wrist
PCK@0.1 at least 85 % (the travel part cannot be measured from single images).

**Limit, to be stated with every number:** COCO is not factory footage. It shows people in everyday scenes, mostly upright,
photographed from many views; the QiSens video shows one worker bending over a bogie, often in a handheld view. So these
numbers say how well the keypoints are found in general, not on QiSens.
