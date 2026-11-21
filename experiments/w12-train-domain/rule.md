# Public train domain test rule (written 2026-11-21, before any run)

**Question.** Does Engine B zero shot, with no tuning, work on railway maintenance video it has never seen, with
another set of classes? Asked by Khurshid after Professor Kim's review. Public data only
(data/public/videos/train_domain, 14 clips of 7 to 60 s, mostly one activity each).

**Code.** Git tag validation-freeze-1 (tools/train_domain.py, analysis/clip_classes.py). The command refuses to run
if the method files differ from the tag, and every result records the tag and the commit.

## Who made what (the split is part of the protocol)

| file | made by | saw the videos |
| --- | --- | --- |
| classes.csv (class, definition) | Gemini, chat 1 | yes |
| labels.csv (file, class, optional second class and change time) | proposed by chat 1, **checked and corrected by Khurshid**; annotators confirm or change each clip on the Labels page | yes |
| descriptions.md (same format as docs_team/phase_descriptions.md) | Gemini, chat 2 | **no, only the class table** |

Claude does not write, edit or see the content of these files before the run, other than the format check.
**Limit:** the class table was made by a model that watched these clips, so the classes fit this set of videos.

## Setting (fixed, untuned)

"B zero shot" of the tag: SigLIP full frame 1, crop 1, VLM 1, temperature 0.01, VLM windows of 3 s with 2 frames,
SigLIP at 1 sample per second, YOLO26 x. All classes are candidates for every clip. Nothing is tuned on these clips,
and each result is reported once, unchanged. A rerun is only allowed after a crash, and is then reported as such.

## a. Clip classification (main test)

- **One label:** the class with the highest total log likelihood over the clip.
- **Two labels:** two segments with two different classes and the best change time, no order given, each segment at
  least 2 s. The decoder is told only that there are two segments.
- **Scored per labelled segment** (a clip with two labels counts as two). For a two label clip, the prediction for a
  true segment is the class covering most of its frames.
- **Reported:** accuracy, per class precision, recall and F1, the confusion matrix, each clip's result.

## b. Joined video (secondary test)

- The 14 clips at 1920x1080, 30 fps, concatenated in the order of a shuffle with seed 20261121. The true boundaries
  come from the clip lengths (and the change time of a two label clip). Neighbouring parts of one class are one segment.
- The decoder gets no order and no number of segments: any class per segment, a class may come back, each segment at
  least 5 s (chosen because the shortest clip is 7.4 s; this uses that one fact about the set). It uses only the class
  likelihood, not the boundary evidence of the ordered decoder.
- **Reported:** frame accuracy, edit, F1@10/25/50, boundary MAE.

## Baselines and limits

- **Engine A is not applicable** to either test: it needs the phase order and has no way to name a class. This replaces
  the Engine A baseline of the first plan. The reference is chance: the share of the most frequent class, and 1 / K.
- 14 clips is a small set: one clip moves the accuracy by about 7 points. The result is reported with its counts and
  not as a general accuracy.
- There is no pass mark. The result is reported as it is, good or bad.
