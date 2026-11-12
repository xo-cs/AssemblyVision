# IKEA ASM macro phases (written 2026-11-12, before any run on IKEA ASM)

**Dataset.** IKEA ASM (Ben-Shabat et al., WACV 2021), `CC BY-NC 4.0`.
- 371 assemblies of 4 furniture types (Lack TV Bench 91, Lack Coffee Table 95, Lack Side Table 95,
  Kallax Shelf Drawer 90).
- The official split has 254 training and 117 testing assemblies.
- We use the top view RGB (camera dev3, 1920x1080, 25 fps).
- The labels are 33 atomic actions plus NA (gt_segments.json, frame ranges at 25 fps).

**Why macro phases.** QiSens phases last minutes. IKEA ASM atomic actions last a few seconds,
and a fine boundary scorer does not transfer to long phases (w10-qisens-engine-b-v0). As for HA4M
(ha4m_macro_phases.md), the actions are grouped into the phases of the assembly. These groups are
defined by the parts being worked on, which the action names state. So the grouping follows the
dataset's own labels, not a judgement per video. The code is `code/assemblyvision/datasets/ikea_phases.py`.

## Grouping

**Lack TV Bench and Lack Coffee Table** (3 phases):

| phase | actions |
| --- | --- |
| Attach legs | pick up leg, align leg screw with table thread, spin leg, tighten leg, lay down leg, rotate table; flip table top, pick up table top, lay down table top, push table top |
| Flip table | flip table, push table |
| Attach shelf | pick up shelf, attach shelf to table, flip shelf, lay down shelf |

**Lack Side Table** (2 phases): Attach legs and Flip table, with the actions above. This table has
no shelf.

**Kallax Shelf Drawer** (4 phases):

| phase | actions |
| --- | --- |
| Side panels | pick up front panel, lay down front panel, pick up side panel, lay down side panel, align side panel holes with front panel dowels, attach drawer side panel |
| Bottom panel | pick up bottom panel, lay down bottom panel, slide bottom of drawer |
| Back panel | pick up back panel, lay down back panel, attach drawer back panel |
| Pins and finish | pick up pin, insert drawer pin, position the drawer right side up |

NA and "other" are don't care.

## Choices made from the label statistics

These use counts of the labels only, no model output.

- **The table top actions join "Attach legs".**
  - As a phase of its own, "Prepare top" would be present in only 30 to 40 % of the Lack videos,
    with a median of 4 s.
  - It always comes right before the legs (or between them).
  - A phase that is mostly absent would be forced into every video by the K phase decoder.
- **"Position the drawer right side up" joins the last Kallax phase.** It is present in 69 % of
  the videos, with a median of 3 s, at the end.
- **The legs are not split into one phase per leg.** The labels do not mark the legs reliably:
  the counts of "pick up leg" and "align leg screw" differ within videos.

## GT rules

These are the HA4M rules (ha4m_phases.macro_index):
- every frame gets the phase of its action;
- don't care frames between two frames of the same phase belong to it;
- between two different phases they are split at the midpoint;
- before the first and after the last action they are "other".

The decoder gets the phase names of the video's furniture type in the order above (K = 2, 3 or 4).

## Statistics (all 371 assemblies)

| furniture | videos | in task order | mean phase shares |
| --- | --- | --- | --- |
| Lack TV Bench | 91 | 71 | legs 78.6 %, flip 6.1 %, shelf 15.3 % |
| Lack Coffee Table | 95 | 75 | legs 71.0 %, flip 7.3 %, shelf 20.6 % |
| Lack Side Table | 95 | 85 | legs 93.5 %, flip 6.5 % |
| Kallax Shelf Drawer | 90 | 37 | side 36.1 %, bottom 22.0 %, back 21.8 %, pins 20.2 % |

"In task order" means each phase appears once, in the order above. In the other videos a phase
comes back (for example legs, flip, legs again) or the Kallax order changes (pins before the back
panel). They are kept and counted, as in HA4M.

## Limits

- **The Lack Side Table is nearly a single phase** (93.5 % legs), so its frame accuracy is easy.
  The selection groups the datasets so that it cannot dominate (rule in w11-visual-engine-b).
- **The phases are shorter than QiSens phases:** tens of seconds, against minutes.
- **The Kallax order varies:** 37 of 90 videos are in task order. The ordered decoder pays for
  that, as it would on QiSens.
