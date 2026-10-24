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
5. Ask before starting any GPU job. GPU0 = GTX 1080 Ti (batch pose),
   GPU1 = RTX A4000 (llama-server VLM and live mode). CUDA_DEVICE_ORDER=PCI_BUS_ID.
6. Never edit .claude/ or this file. Khurshid owns the rules.

## Environment
- Always: source ~/AssemblyVision/av.sh (conda env ~/AssemblyVision/env, Python 3.11)
- torch 2.14.1+cu126, ultralytics 8.4.178, opencv 5.0.0
- Ubuntu 20.04, gcc 7.5, nvcc 11.3, cmake 3.16 (old system toolchain)
- Folders: code/ (assemblyvision package, LLD 3.0.2), docs/ HLD and LLD, models/ weights, logs/ logs of public runs, data/ (see layout)

## Writing style for documents and comments
No hyphens or em dashes in prose; use parentheses or commas instead.
