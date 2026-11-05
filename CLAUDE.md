# AssemblyVision (Team 4, QiSens capstone)

AI motion analysis for train assembly workers: YOLO26 pose, ByteTrack, step
segmentation (Engine A rules, Engine B MS TCN), motion metrics, local VLM
suggestions (Qwen3 VL in llama-server), Gradio UI, PDF report.
Design reference: HLD v1.1 and LLD v1.0 (error codes AV-XXX-nnn, tests TC-XXX-nn).

## Data layout
- data/qisens/ : QiSens video plus EVERYTHING derived from it (runs, frames,
  keypoints, overlays, heatmaps, reports, logs, temp files, av.db). FORBIDDEN.
- data/public/ : public videos and datasets (InHARD, HA4M, HA ViD, IMPACT,
  bogie video, ...), their runs and av_dev.db. Free to read and use.

## Hard rules (never break these)
1. Never open, read, list contents of, decode or print anything under
   data/qisens/. Your model runs in the cloud and this data must never leave
   the server.
2. Never run any script or command against data/qisens/. Khurshid runs every
   QiSens job himself and shares only tracebacks or timings.
3. Code must route ALL output of a QiSens run (including temp frames and logs)
   under data/qisens/, and use av_dev.db under data/public/ for public runs.
4. Everything stays inside ~/AssemblyVision. No sudo, no apt, no system wide
   installs, no changes outside this folder. Do not break the server.
5. Ask before starting any GPU job. GPU0 = RTX A5000 24 GB (Ampere, bus 01:00,
   batch pose, YOLO26 x in FP32 by default; pose.half = false, true asks for FP16),
   GPU1 = RTX A4000 16 GB (Ampere, bus 03:00, llama-server VLM and live mode).
   One job per GPU (.locks/gpu<N>.lock). CUDA_DEVICE_ORDER=PCI_BUS_ID.
6. Never edit .claude/. Edit this file only when Khurshid asks. Khurshid owns the rules.
7. Never read or change anything of the qisens1 account (Dr. Choi's own copy of
   the project). Work only in ~/AssemblyVision of this account.
8. GitHub: push only to the remote git@github-av:xo-cs/AssemblyVision.git. Never
   force push, never rewrite history, never push tags unless Khurshid asks. Push
   only after the pre-push check (.git/hooks/pre-push) passes. Push when Khurshid
   asks, or at the end of a task he approved, and say so in the report.

## Environment
- Always: source ~/AssemblyVision/av.sh (conda env ~/AssemblyVision/env, Python 3.11)
- torch 2.14.1+cu126 (supports both Ampere GPUs; do not upgrade), ultralytics 8.4.178, opencv 5.0.0
- models/ holds YOLO26 pose n, s, m, l and x (Ultralytics assets v8.4.0, checksums in
  models/SHA256SUMS_yolo26_pose.txt); x is the batch default
- Ubuntu 20.04, gcc 7.5, nvcc 11.3, cmake 3.16 (old system toolchain)
- Folders: code/ (assemblyvision package, LLD 3.0.2), docs/ HLD and LLD, models/ weights, logs/ logs of public runs, data/ (see layout)

## Experiments
The project will produce a paper. Every experiment (model, precision, tracker,
engine, setting) runs with a stored config on the same data, and its results are
saved as a table under experiments/ (tracked in git), so the paper's tables can be
regenerated. Public data only; QiSens results go there only as summary numbers
Khurshid approves.

## Writing style for documents and comments
No hyphens or em dashes in prose; use parentheses or commas instead.
