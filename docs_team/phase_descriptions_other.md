# "Other" class descriptions for zero shot phase scoring (written 2026-11-13, before any run with them)

**Status.** Frozen when committed, before any run of seg_a.other_class on public or QiSens data. Never
edited after a result; a change needs a new file and a new rule.

**Purpose.** seg_a.other_class adds one extra class to the zero shot scorers (SigLIP 2 and Qwen3 VL):
no work in progress. The DP may use it only before the first phase and after the last (an intro or an
outro). The five QiSens phase descriptions (phase_descriptions.md) and the IKEA ones
(phase_descriptions_ikea_asm.md) stay frozen and unchanged; this is a separate file.

**Sources.** General knowledge of how such recordings start and end: title or text cards, people
standing, talking or walking, an empty workshop or room. No QiSens frame, video or annotation was used.

**Format** (read by code/assemblyvision/analysis/phase_text.py): the same as the phase files, one
`# Furniture:` group per data set ("QiSens" and "IKEA ASM"), each with one `## ` class, a
`Distinguishes:` line, three `> ` lines for the VLM prompt and `- ` lines for SigLIP.

# Furniture: QiSens

## No maintenance work

Distinguishes: nobody is working on the vehicle or a part yet or any more: people only stand, talk, walk or wait, or the picture is a title or text card or an empty workshop (against all five phases, where hands handle tools or parts).

> No maintenance work is going on: no hands on tools, parts or the vehicle.
> Typical: a title or text card, people standing and talking, a worker walking in or out of the picture, an empty workshop.
> It can only come before the work starts or after it has ended.

- a title card with white text on a dark background
- a text slide with the name of a maintenance procedure
- a black screen with a caption
- a logo and a title shown before a training video
- an empty railway workshop with a bogie and nobody near it
- an empty workshop floor seen by a fixed CCTV camera
- a parked train in a depot with nobody working on it
- two workers standing in a workshop and talking to each other
- a worker standing still with empty hands, listening to an instructor
- a group of people standing beside a railway vehicle, having a briefing
- a worker in safety clothes walking across the workshop floor
- a worker walking into the picture towards the bogie
- a worker walking away and leaving the picture
- a person putting on gloves and a helmet before work
- a worker standing with arms crossed, waiting
- a person pointing at the vehicle while explaining something
- a worker looking at a tablet or a sheet of paper, not touching any part
- a wide CCTV view of a quiet workshop with people standing at the side
- a wide view of a depot where people walk past a train without working on it
- a close view of a person's face talking towards the camera
- a close view of empty hands resting, no tool and no part
- a camera being set up, the picture shaking, a floor or a wall in view
- the end of a video, a person waving or turning away from the camera

# Furniture: IKEA ASM

## No assembly work

Distinguishes: nobody is assembling yet or any more: the person only stands, walks, unpacks nothing and touches no panel, or the room is empty (against every assembly phase, where hands handle panels, legs, pins or screws).

> No assembly work is going on: no hands on furniture parts.
> Typical: the person walks into or out of the view, stands and looks around, waits, or the room is empty.
> It can only come before the assembly starts or after it has ended.

- an empty room seen from above with furniture parts lying on the floor
- a top view of a floor with a rug and nobody in the picture
- a person walking into the room seen from above
- a person walking out of the view from above
- a person standing still next to flat packed furniture parts
- a person standing with empty hands, looking at the camera
- a person standing and looking around the room
- a person reading an instruction sheet, not touching any part
- a person talking, arms hanging down, no part in the hands
- a person stretching or resting after the work
- a finished piece of furniture standing in an empty room
- a person stepping back to look at the finished furniture
- a top view of a person waiting next to a table top lying on the floor
- a person bending over a box but not taking any part out
- a person sitting on the floor resting with empty hands
- a person adjusting the camera, the picture moving
- a top view of feet walking across the floor
- a person waving at the camera at the end of the recording
- a top view of an empty floor at the end of a recording
- a person standing up from the floor and walking away
- a quiet room with furniture parts laid out and nobody moving
