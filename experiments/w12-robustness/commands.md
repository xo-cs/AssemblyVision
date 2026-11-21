# w12-robustness: commands (Khurshid runs them on QiSens)

```
cd ~/AssemblyVision/code && source ~/AssemblyVision/av.sh
python -m assemblyvision.tools.freeze --check
```

**1. Pose model (group e), CPU.** Makes a hidden copy of the l run with its own SigLIP scores (about 1,200 samples, two views, so it can take a while on the CPU) and prints its run ID:
```
python -m assemblyvision.tools.robustness prepare-pose --l-run ../data/qisens/runs/run_20261025_053609_fd62 --x-run ../data/qisens/runs/run_20261108_133251_1550
```

**2. VLM sampling (group d), GPU1, only after your 4D redo has finished:**
```
python -m assemblyvision.tools.vlm_scores --run ../data/qisens/runs/run_20261108_133251_1550 --gpu 1 --window-s 3 --frames 2 --out vlm_3s2f.npz
```

**3. All groups, CPU.** Replace `L_COPY` with the run ID from step 1. Group c counts the files that exist in docs_team/independent_descriptions; a group without its input is reported as not tested, so run this again when the three author files are in:
```
python -m assemblyvision.tools.robustness run --run ../data/qisens/runs/run_20261108_133251_1550 --gt ../data/qisens/annotations/steps_gt.csv --reference 93.20 --vlm-file vlm_3s2f.npz --l-run ../data/qisens/runs/L_COPY
```

Output: data/qisens/validation/robustness.{json,md}. The printed tables hold summary numbers only. Paste them back once you approve them, and they go into experiments/w12-robustness/table.md unchanged.
