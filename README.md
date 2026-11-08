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
docs_team/             team documents (labelling guide)
scripts/               helper scripts (Label Studio launcher)
```

Not in git: `data/`, `models/` (weights), `env/` (conda env), `docs/` (HLD and LLD),
`logs/`, `tmp/`, local tools and caches.

## Environment

Ubuntu 20.04, conda env with Python 3.11, torch 2.14.1+cu126, ultralytics 8.4.178,
opencv 5.0.0. Everything lives inside `~/AssemblyVision`. The exact packages of the env are pinned in
`code/requirements-pinned.txt`.

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
tracking, the Engine A steps and the motion metrics are unchanged. The weaker verdict
"equivalent within numerical noise" allows differences no larger than the measured noise
floor of TF32 against strict FP32 at batch 1 (`exp noise-floor`, `experiments/noise_floor.json`;
the rule is at the top of `experiments/INDEX.md`; metrics above the floor are listed with their
absolute values). The default profile is the w10-best-b8 setting (pose batch 8, TF32), which is
above that floor with identical Engine A steps, NULL decisions and flags; accuracy checks use
`--config configs/reference.toml` (batch 1, strict FP32; `exp new` takes `--config` too).
`shots.parallel` (off by default) runs shot detection in a thread while the pose pass starts.
`tools/run_diff RUN_A RUN_B` compares two runs by their manifests only (settings, model, shots,
subject choice, GT checksum). Public experiments are
tracked in git. A QiSens experiment (`--source qisens`) keeps its config and results under
`data/qisens/experiments/` only; just an approved summary row is added to `experiments/`.

## Public datasets for Engine B

Engine B (a taxonomy free boundary scorer) is pretrained on public data, using keypoints from our
own pose model, so public and partner keypoints share model and format. Datasets live under
`data/public/datasets/<name>/` only; the split lists and the joint mapping report are tracked in
`experiments/datasets/`.

```bash
python -m assemblyvision.tools.fetch_dataset ha4m listing          # Science Data Bank V2 listing (217 zips, 3.2 TB)
python -m assemblyvision.tools.fetch_dataset ha4m annotations      # Labels.txt + skeletons of every video (about 3 GB)
python -m assemblyvision.tools.engine_b_data ha4m                  # Engine B cache (extends as videos arrive)
python -m assemblyvision.tools.dataset_splits ha4m subjects        # fixed cross subject split of the 41 subjects (once)
python -m assemblyvision.tools.dataset_splits ha4m splits          # cross subject and cross location video lists
python -m assemblyvision.tools.ha4m_domain_gap fetch|run|compare   # YOLO26 x against the Kinect joints (run = GPU job)
```

Engine B trains on HA4M's own skeletons: the Kinect 2D colour joints mapped to COCO 17, in the
body frame of the reach heatmap (mid hip origin, hip to shoulder axis up, torso lengths), with an
augmentation that makes them look like YOLO26 output (`[engine_b]`: joint jitter, joint and frame
dropout, confidence levels and noise). The RGB route (our pose on the dataset videos,
`fetch_dataset ha4m rgb` and `tools/dataset_pose`) is kept but not used: the server gives about
0.2 MB/s. The domain gap experiment measures how far YOLO26 x is from the Kinect joints on a short
window of frames per lab. The download reads only the wanted members of each remote zip (HTTP range
requests), one paced stream, honouring HTTP 429; a transfer above 5 GB needs `--yes`; a stopped
download resumes per video (`scripts/ha4m_download.sh` in the tmux window `dl`). InHARD has a loader
interface and a BVH reader; its data is not downloaded until access is granted.

The model (`analysis/segmentation/boundary_tcn.py`) is a 2 stage MS TCN on the 55 features at
10 fps that gives a boundary probability per frame (targets: Gaussian bumps of sigma
`boundaries.tol_s` on every action change, weighted BCE). In Engine A it is one more evidence
term of the DP, off by default (`seg_a.engine_b_weight = 0`), pooled over `engine_b.pool_s` (30 s, chosen
on public runs in w10-engine-b-pool together with the weight 0.4 to try):

`engine_b.hip_calibration` (off by default) first moves every Kinect joint by the mean YOLO26 x
offset of that joint over both labs (w10-ha4m-domain-gap). `--folds` adds the cross location folds
(one model each). `engine_b_firing` gives the peaks per minute against the true change rate per pool
window; the compare table shows the same rate on a run ("B peaks/min raw/pooled").

```bash
python -m assemblyvision.tools.train_engine_b --id w10-engine-b-v0 --gpu 0 \
    --folds cross_subject,cross_location_test_lab_a,cross_location_test_lab_b   # GPU job
python -m assemblyvision.tools.engine_b_firing --id w10-engine-b-firing --model w10-engine-b-v0   # CPU
python -m assemblyvision.tools.engine_b_grid --id w10-engine-b-pool --model engine_b/w10-engine-b-v0.pt \
    --run ../data/public/runs/RUN --gt ../data/public/videos/CLIP_gt.csv     # CPU, public runs only
```

Engine B v1 (`engine_b.action_weight > 0`) adds the 12 HA4M action head (multitask). In Engine A
its embedding (the last stage's unit length hidden features) gives a phase level change score:
the distance between the mean embedding of the `seg_a.embed_window_s` before and after a time,
weighted by `seg_a.embed_weight` (default 0). `seg_a.start_detect` (default off) starts phase 1
at the first sustained subject activity, so an intro becomes "other".

```bash
python -m assemblyvision.tools.train_engine_b --id w10-engine-b-v1 --gpu 0 --set engine_b.action_weight=1.0 \
    --folds cross_subject,cross_location_test_lab_a,cross_location_test_lab_b              # GPU job
python -m assemblyvision.tools.engine_b_grid --id w10-engine-b-embed --model engine_b/w10-engine-b-v1.pt \
    --weight-key seg_a.embed_weight --window-key seg_a.embed_window_s --run ... --gt ...      # CPU
python -m assemblyvision.tools.run_pose --run RUN_DIR --gt GT_CSV --compare seg_a.start_detect
python -m assemblyvision.tools.run_pose --run RUN_DIR --gt GT_CSV --compare seg_a.engine_b_weight \
    --set engine_b.model='"engine_b/w10-engine-b-v0.pt"'                       # CPU, writes nothing
python -m assemblyvision.tools.run_pose --run RUN_DIR --gt GT_CSV --compare engine_b.pool_s=0,10,30,60 \
    --set seg_a.engine_b_weight=0.4 --set engine_b.model='"engine_b/w10-engine-b-v0.pt"'
```

## Pose labelling

Ground truth for the detection precision and recall, OKS AP and PCK of our pose model comes from a
local Label Studio (its own venv in `envs/labelstudio`, analytics off, 127.0.0.1 only) and a frame
server; the team labels through an SSH tunnel. See `docs_team/labelling_guide.md`.

```bash
python -m assemblyvision.tools.sample_frames frames --run <run_dir> --n 200    # 150 to 300 stratified frames
scripts/labelstudio.sh start <public|qisens>                                   # Label Studio on 127.0.0.1:8080
python -m assemblyvision.tools.label_frames_server --run <run_dir>             # frames on 127.0.0.1:8081
python -m assemblyvision.tools.sample_frames crops --run <run_dir> --boxes <box_export.json>
python -m assemblyvision.tools.eval_pose_labels --run <run_dir> --boxes <box_export.json> \
    --keypoints <keypoint_export.json> --labellers 1=khurshid,2=asel,3=kazi
```

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
