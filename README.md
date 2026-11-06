# AssemblyVision

AssemblyVision (Team 4, QiSens capstone) is an AI motion analysis system for
train assembly and maintenance workers. From a recorded video it finds the
workers' body keypoints, follows each worker over time, splits the work into
process steps, measures how the body and hands move in each step, and gives a
documented baseline of the current process with stated accuracy. A locally
hosted vision language model adds a limited set of improvement suggestions that
the user accepts or rejects. Everything runs on one server; no video and no
result leaves it.

## Pipeline

1. Ingest: check the video, convert it to constant frame rate, split it into
   shots (fixed camera views).
2. Pose: YOLO26 pose (n, s, m, l or x) gives 17 body keypoints per person and frame.
3. Tracking: ByteTrack based tracker (KTS v1) keeps one identity per worker.
4. Cleaning: gap filling and smoothing of the keypoint tracks.
5. Step segmentation: Engine A (rules over motion and activity change), giving
   step boundaries and durations; Engine B (MS TCN, pretrained on public data) is
   planned, and the paper compares both engines on the same footage.
6. Motion metrics: per step durations against a cycle time baseline, travel,
   hand reach, trunk posture and other ergonomic measures.
7. Suggestions (planned): Qwen3 VL 8B in a local llama-server, on a few sampled frames per step.
8. Output: skeleton and track overlay videos, timeline, heatmaps, cycle time
   graph and a Gradio interface (New analysis, Results, Runs, Live); the PDF
   report is planned.

Live mode runs a lighter, online version of steps 2 to 6 on a camera stream.

## Repository layout

```
code/assemblyvision/   the Python package (ingest, perception, analysis, live, output, storage, runs, ui, tools)
code/configs/          default.toml (all settings), steps.toml (step vocabulary), live/ station presets
code/tests/            unit tests (unittest)
av.sh                  session setup (conda env, caches, CUDA_DEVICE_ORDER)
CLAUDE.md              working rules for the project and its coding assistant
```

Not in git: `data/`, `models/` (weights), `env/` (conda env), `docs/` (HLD and LLD),
`logs/`, `tmp/`, local tools and caches.

## Environment

Ubuntu 20.04, conda env with Python 3.11, torch 2.14.1+cu126, ultralytics 8.4.178,
opencv 5.0.0. Everything lives inside `~/AssemblyVision`.

```bash
source ~/AssemblyVision/av.sh          # activates ~/AssemblyVision/env, sets caches and CUDA_DEVICE_ORDER=PCI_BUS_ID
cd ~/AssemblyVision/code
```

Model weights go in `~/AssemblyVision/models/` (YOLO26 pose n, s, m, l, x from the
Ultralytics assets release, checked against `SHA256SUMS_yolo26_pose.txt`; Qwen3 VL 8B
GGUF files for llama-server).

## Running

Batch analysis of one video (ingest, pose, tracking, segmentation, metrics, overlay):

```bash
python -m assemblyvision.tools.run_pose --source public --video ../data/public/videos/<video>.mp4 \
    --variant x --gpu 0 [--gt <steps_gt.csv>]
```

Rerun later stages of an existing run on the CPU, from its stored files:

```bash
python -m assemblyvision.tools.run_pose --run <run_dir> --stages segmentation,metrics,overlay,eval [--gt <steps_gt.csv>]
```

The interface (binds 127.0.0.1 only, default port 7860):

```bash
python -m assemblyvision.ui.app --sources public
# on the laptop: ssh -L 7860:127.0.0.1:7860 <user>@<server>, then open http://127.0.0.1:7860
```

Tests (CPU):

```bash
python -m unittest discover -s tests
```

A full run on a GPU takes the lock `.locks/gpu<N>.lock`, so only one job runs per GPU.

## Experiments (paper tables)

Every result in the paper comes from a registered experiment: `experiments/<id>/` holds
`config.toml` (every setting), `results.json` (git commit, model checksum, GPU, driver,
library versions, data area, date, every repeat, medians, GPU cost, equivalence with the
reference) and `table.md`. `experiments/INDEX.md` lists them, one line each.

```bash
python -m assemblyvision.tools.exp new w10-b16 --from w10-ref --reference w10-ref \
    --video ../data/public/videos/<video>.mp4 --set pose.batch=16 --repeats 3
python -m assemblyvision.tools.exp run w10-b16          # one GPU job; refused if another process uses the GPU
python -m assemblyvision.tools.exp table w10-ref w10-b16
python -m assemblyvision.tools.exp recheck w10-b16       # CPU: compare again after the reference was rerun
```

Each repeat is a separate full run with the stage profiler on (per stage times, the pose
breakdown, GPU utilisation and memory from nvidia-smi). A speed variant counts as
equivalent only if the pose output stays within 1 px (confidences within 0.01) and the
tracking, the Engine A steps and the motion metrics are unchanged. Public experiments are
tracked in git. A QiSens experiment (`--source qisens`) keeps its config and results under
`data/qisens/experiments/` only; just an approved summary row is added to `experiments/`.

## Data layout

| Folder | Content | In git |
| --- | --- | --- |
| `data/qisens/` | The partner's footage and everything derived from it (runs, frames, keypoints, overlays, reports, logs, temp files, `av.db`). Private: it never leaves the server and is never committed. | never |
| `data/public/` | Open videos and datasets (InHARD, HA4M, HA ViD, IMPACT and others), their runs and `av_dev.db`. Used for development, tests and pretraining. | no (too large) |

The code routes every output of a QiSens run, temp files and logs included, under
`data/qisens/`, and the interface shows a QiSens video only by a neutral name.
`.gitignore` and a git pre push hook keep data, weights, logs and media files out of git.

## Hardware

| GPU | Card | Use |
| --- | --- | --- |
| GPU0 | NVIDIA RTX A5000 24 GB | batch pose (YOLO26 x, FP32 by default) |
| GPU1 | NVIDIA RTX A4000 16 GB | live mode and the llama-server VLM |

## Team

AssemblyVision, Team 4

| Member | Role |
| --- | --- |
| Ortikov Khurshid Golibovich | Team leader, pose estimation and model lead |
| Shahid Huzaifa | Co leader, motion analysis and behavior engineer |
| Alybaeva Asel | Gradio app and frontend developer, meeting notes and report writer, presentation maker |
| Opurbo Kazi Shahmiran | Data pipeline and server infra engineer, project progress organizer, presentation maker |

Mentor: CEO Choi Jin Wook (QiSens). Professor: Prof. Edward Youngil Kim (Woosong University).

Design documents: HLD v1.1 and LLD v1.0 (shared separately).
