# Public train domain test: protocol

Rule: experiments/w12-train-domain/rule.md (fixed before any run). Results: experiments/w12-train-domain/table.md.
Code: git tag validation-freeze-1.

**Status (2026-11-26).** Run once on tag validation-freeze-1: **13 of 14 clips correct (92.86 %); joined video frame accuracy 89.69**. Results and their reading: experiments/w12-train-domain/table.md.

**How the labels were made (part of the protocol):**
- **One annotator.** Khurshid confirmed or set the class of all 14 clips on the Labels page
  (annotations/khurshid.csv). labels.csv is that file. There is no second annotator, so no agreement can be reported.
- **7 of the 14 clips had no Gemini draft**, because the draft named them video_1.mp4 to video_7.mp4, which do not exist.
  Khurshid labelled those 7 alone, without a draft.
- For the other 7, the Gemini draft and Khurshid's label agree on all 7.
- No clip has a second class.
- Class counts: Track and Ballast Maintenance 6, Equipment and Machinery Coordination 2, Washing and Cleaning 2, Locomotive
  Firing and Fueling 2, Rolling Stock and Coupler Work 1, Metal Fabrication and Grinding 1. Chance: 1 / 6 = 16.7 %, the
  most frequent class 6 / 14 = 42.9 %.

**Files, with the changes made:** format fixes only (labels.csv header change_time to change_s, "* " bullets to "- ",
"Differs from:" to "Distinguishes:", final newlines). descriptions_short.md (3 "> " lines per class, from the same second
Gemini chat, which saw only the class table) was merged under each class heading of descriptions.md. Every original is
kept in originals/.

## Files in data/public/videos/train_domain/

| file | content | from |
| --- | --- | --- |
| 14 clips (*.mp4) | public railway maintenance video | Khurshid |
| classes.csv | `class,definition` | Gemini chat 1 (watched the videos) |
| labels.csv | `file,class,second_class,change_s` (second_class and change_s empty for one label) | chat 1, checked by Khurshid |
| descriptions.md | `## class` in the order of classes.csv, three `> ` lines, then `- ` lines | Gemini chat 2 (**only the class table**) |
| *.template | the empty formats | made by the tool |
| annotations/<user>.csv | each annotator's confirmed or changed classes | Labels page |
| generated/ | run list, joined video, run settings | the run command |

## Steps

1. Put classes.csv in the folder. Give **only** that table to a new Gemini chat to write descriptions.md.
2. Annotators open **Labels** in the app, choose *Train domain clips*, and confirm or change each clip's class.
3. Khurshid makes the final labels.csv.
4. Check the files, then run (needs GPU0 and GPU1 free; it refuses if a GPU lock is held):

```
cd ~/AssemblyVision/code && source ~/AssemblyVision/av.sh
python -m assemblyvision.tools.train_domain check
python -m assemblyvision.tools.train_domain run --gpu 0
```

The command runs the 14 clips and the joined video, evaluates them, and writes experiments/w12-train-domain/results.json
and table.md. If it stops, the same command continues from the runs already done.
