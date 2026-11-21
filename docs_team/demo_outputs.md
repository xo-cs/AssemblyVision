# Demo outputs: end to end check

Checked with tools/check_outputs.py, which builds each output through the code the app and the report use and changes
nothing in the run. Result on a public demo run (run_20261120_075124_83fe), at tag validation-freeze-1. **The flagship
has not been checked yet:** that needs the command below.

| output | status | note |
| --- | --- | --- |
| process timeline | works | |
| skeleton overlay video | works | |
| movement heatmap | works | GVHMR floor map in metres, and the 2D hand reach map |
| cycle time graph | **partial** | a table of median and quartiles per phase; there is no graph |
| improvement report (PDF): measured baseline | works | timeline, steps, metrics, notices, provenance |
| improvement report (PDF): labelled suggestions | **missing** | no section, and no code generates suggestions |

**Also missing for the demo:** the movement path in metres is not in metrics.csv or the PDF (only in this check). The
ordered decoder finds each phase once per video, so the cycle baseline has one cycle per phase; a real spread of cycle
times needs repeated cycles.

These gaps are listed and not built, because feature work is stopped. Whether to build the cycle graph and the
suggestions section is Khurshid's decision.

**Flagship (Khurshid runs it; prints status and path lengths only):**

```
cd ~/AssemblyVision/code && source ~/AssemblyVision/av.sh
python -m assemblyvision.tools.check_outputs --run ../data/qisens/runs/run_20261108_133251_1550
```

Run it after the 4D redo. Exit code 3 means at least one output is not "ok"; the two gaps above always give it until
they are built.
