# Public train domain test: protocol

Rule: experiments/w12-train-domain/rule.md (fixed before any run). Results: experiments/w12-train-domain/table.md.
Code: git tag validation-freeze-1.

**Status.** Folder, templates, label page and command are ready. **The test has not been run**, and it is not run
until Khurshid says the Gemini files are in.

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
