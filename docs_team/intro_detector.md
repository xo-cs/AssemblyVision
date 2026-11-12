# Motion energy intro and outro detector (design note, 2026-11-12)

Status: **design only, not built.** It is built after Khurshid's review.

## 1. Why a separate detector

- **Item 8 failed as a DP option** (w10-other-ends-cv, -test). The intro and outro were edge
  phases inside the DP, with a "no intro" edge candidate of strength `other_min_strength`.
  - The DP maximises the total strength, so with dense candidates it always prefers a strong
    change score point for the intro and outro boundaries.
  - Every bias below 1 lowered the frame accuracy, even on videos without an intro (89.35 to
    33.03). A bias of 1 never fires.
- **An intro costs a lot when nobody marks it:** a still intro of 5 to 30 s lowers v2 from 88.53
  to 72.77 on the HA4M test split.
- **So the decision "is there an intro, and where does work start" has to be made before the DP,
  from evidence that is about the intro itself.** The worker is present but not working: little
  hand motion, little whole body motion.
- **The existing `seg_a.start_detect`** (a sustained share of moving frames from the first person
  frame) is the one sided version of this idea. It had no effect on QiSens (w10-qisens-steps-drop),
  for two reasons:
  - its threshold is the absolute `pause_energy`, not the video's own working level;
  - it does not look at the outro.

## 2. Detector

Signals per frame (all exist already in the cleaned series):

- `energy`: the subject's motion energy;
- `hand speed`: wrist speed in torso lengths per second;
- `present`: a subject track exists.

**Working level:** the median of energy over the frames with a subject, and its spread (the
median absolute deviation). Both are measured per video, so camera distance, frame rate and pose
model do not set the threshold.

**Active frame:** energy or hand speed above `working level - k * spread` (k = `intro.k`, about 1),
smoothed by a moving median of 2 s.

**Start of work:**
- the first frame t from which at least `intro.active_share` (0.6) of the frames in the next
  `intro.sustain_s` (10 s) are active;
- **hysteresis:** after that, a pause shorter than `intro.max_gap_s` (5 s) does not end the work.

**End of work:** the same, run backwards from the last subject frame.

**Limits:**
- an intro or outro must be at least `intro.min_s` (3 s) and at most `intro.max_share` (0.25) of
  the active span;
- otherwise it is not an intro, and nothing changes.

**Output:** the active span for the DP becomes [start of work, end of work). The intro and outro
become `other`. The DP, its limits (computed on the shorter span) and the candidates are
unchanged.

**Report:** start and end of work, the working level, the share of active frames, and "found" or
the reason for "not found". This goes in the segmentation report, like start_detect today.

## 3. Evaluation (rule written before any run)

- **Train split:**
  - the HA4M benchmark videos, plain and with synthetic still intros and outros (5 to 30 s,
    seeded), as in item 8;
  - in addition, intros with small motion (a worker who walks in or arranges parts: the first
    seconds of the video replayed with jitter), since a perfectly still intro is too easy.
- **Grid:** `intro.k` 0.5, 1, 1.5; `intro.sustain_s` 5, 10, 20; `intro.active_share` 0.5, 0.6,
  0.8. Chosen by CV.
- **Criterion:** mean frame accuracy over plain and intro videos.
- **Gate:** the detector is kept only if it changes nothing on the plain videos (no false intros:
  the plain frame accuracy stays within 0.5 points).
- **Test split once,** and the two public clips: process_480p30 must stay at 92.33.
- **QiSens:** one blind compare against the current default.
- **Default:** off, until Khurshid decides.

## 4. Open questions for Khurshid

- Does the QiSens video have an intro or outro of this kind (a worker present but not working)? Its
  GT would tell. Khurshid can answer with yes or no and an approximate length, without numbers
  from the GT.
- Should a walk in (the worker enters the view) count as intro, or should only no_person time
  before the first track do so (already other today)?
