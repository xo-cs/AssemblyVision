# Demo outputs: end to end check

Checked with tools/check_outputs.py, which builds each output through the code the app and the report use and changes
nothing in the run. Result on a public demo run (run_20261120_075124_83fe). **The flagship has not been checked yet:**
that needs the commands below.

| output | status | note |
| --- | --- | --- |
| process timeline | works | |
| skeleton overlay video | works | |
| movement heatmap | works | GVHMR floor map in metres, and the 2D hand reach map |
| cycle time graph | works | Results and PDF; one bar per cycle when a phase repeats |
| movement path in metres | works, **not validated** | per phase in Results, the PDF and the report's metrics |
| tool pickup and object placement | runs, **experimental** | **0 events on 3 public runs**; the COCO detector has almost no workshop tools |
| improvement report (PDF): measured baseline | works | timeline, steps, metrics, notices, provenance |
| improvement report (PDF): suggestions | works, **experimental** | rule based, each with metric, value, threshold and time ranges, labelled as suggestions and not findings |

**Known limits for the demo.** The ordered decoder finds each phase once per video, so the graph has one cycle per phase
and shows no spread of cycle times. The suggestion thresholds are team defaults, not ergonomic limits. Do not present a
count of 0 tool pickups as a result: it means the detector did not see the tools.

**Flagship (Khurshid runs them, CPU only; they print counts, status and path lengths, no frames):**

```
cd ~/AssemblyVision/code && source ~/AssemblyVision/av.sh
python -m assemblyvision.tools.detect_objects --run ../data/qisens/runs/run_20261108_133251_1550
python -m assemblyvision.tools.check_outputs --run ../data/qisens/runs/run_20261108_133251_1550
```

Exit code 3 means at least one output is not "ok".
