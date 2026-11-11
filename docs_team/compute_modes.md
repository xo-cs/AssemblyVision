# Compute modes (design, for review before the dual GPU code)

Status: approved 2026-11-12 with the additions below (sections 3, 4, 6 and 8). It is implemented only
after the segmentation items of that day, as the lowest priority. gpu_single is today's behaviour.

## 1. Modes

| mode | when | devices | output |
| --- | --- | --- | --- |
| cpu_only | no CUDA device, or `compute.mode = "cpu_only"` | CPU | same files; warning AV-RUN-008 with an ETA |
| gpu_single | today's path; the default | one GPU (GPU0 for batch runs) | as now |
| gpu_dual | `compute.mode = "auto"` or `"gpu_dual"`, and the GPU1 conditions of section 4 hold | GPU0 and GPU1, one process each | DET files merged in frame order, then tracking as now |

New settings (section `[compute]`, all declared in experiments/noop.py):

- `mode = "gpu_single"`: one of `"auto"`, `"cpu_only"`, `"gpu_single"` or `"gpu_dual"`. The
  default stays gpu_single until the validation of section 6 passes. `"auto"` lets the planner
  choose; `"gpu_dual"` means dual if possible, else single, with the reason recorded.
- `dual_min_frames = 3000`: shorter videos stay single, because the start up cost (a second model
  load and decoder) outweighs the gain.
- `chunk_max_s = 30.0`: the longest chunk, in seconds of video.
- `speed_file = "configs/gpu_speed.toml"`: the measured pose fps per GPU, variant and batch.

## 2. Device planner

`runs/planner.py`, called once by run_pose before any stage, on the CPU.

1. **Read the state** without taking anything: CUDA devices from nvidia-smi (index, bus, name,
   free memory, compute processes), the locks `.locks/gpu<N>.lock` (held or not, read from
   /proc/locks as views.py does), and the live session marker (section 4).
2. **Choose the mode:**
   - `cpu_only` if no CUDA device exists, or if it is asked for;
   - `gpu_dual` if it is asked for (or `auto`) and every condition of section 4 holds and the
     video has at least `dual_min_frames` frames;
   - else `gpu_single` on `gpu.pose` (GPU0), waiting (queued) for its lock exactly as today.

   The waiting rule does not change: a busy GPU0 means wait, never a silent CPU fallback. A CPU run
   of a 10 minute video takes hours.
3. **Take the locks** in a fixed order (gpu0, then gpu1), so two planners can never deadlock. GPU1 is
   taken without waiting; if it is busy at that moment the plan becomes gpu_single.
4. **Record the plan** in `manifest["compute_plan"]` before the first stage (example values):
   ```json
   {"mode": "gpu_dual", "requested": "auto", "reason": "GPU1 idle, no live session, 17998 frames",
    "devices": [{"gpu": 0, "bus": "01:00.0", "name": "RTX A5000", "lock": ".locks/gpu0.lock",
                 "share": 0.64, "chunks": [0, 11], "frames": 11520},
                {"gpu": 1, "bus": "03:00.0", "name": "RTX A4000", "lock": ".locks/gpu1.lock",
                 "share": 0.36, "chunks": [12, 19], "frames": 6478}],
    "speed_source": "configs/gpu_speed.toml sha256 ...", "layout_sha256": "...",
    "preempted": null}
   ```
   For cpu_only the record holds the ETA and the measured CPU fps. Every gpu_single run records
   `{"mode": "gpu_single", "devices": [{"gpu": 0, ...}]}`, so all runs are comparable.

## 3. gpu_dual: the split

The split must not depend on timing, so the same video and configuration always give the same
layout.

- **Shot table first.** The chunks follow the shot cuts, so `shots.parallel` does not apply in dual
  mode (recorded as `"shots_parallel": "off in gpu_dual"`).
- **Chunk layout:**
  - every shot is one chunk if it is at most `chunk_max_s` long; a longer shot is cut into
    equal parts of at most `chunk_max_s`, at fixed frame numbers;
  - no chunk boundary falls inside a shot's reference search window (the first `ref_search_s`), so
    every reference frame is found by the process that owns the shot start;
  - the layout depends only on the shot table, the fps and the two settings, and its sha256 is in
    the plan.
- **Assignment:**
  - shares come from the stored speeds, not from a measurement in this run:
    `share0 = v0 / (v0 + v1)` for the run's variant and batch, from `compute.speed_file`. That file
    is written by a benchmark (bench_pose on both GPUs) and committed, so a run never times anything
    to decide the split;
  - GPU0 takes chunks 0..k, where k is the first chunk at which the cumulative frames reach
    `share0` of the total; GPU1 takes the rest;
  - two contiguous ranges keep decoding sequential, since each process seeks once to its first
    frame.
- **Processes:**
  - one pose process per GPU (CUDA_VISIBLE_DEVICES=<index>, CUDA_DEVICE_ORDER=PCI_BUS_ID), each with
    the run's full configuration, writing DET parts `det/<shot>.part<chunk>.npz` and its log under
    the run folder. For a QiSens run that is under data/qisens (hard rule 3);
  - the parent waits for both processes.
- **Merge:**
  - per shot, the parts are concatenated in frame order into the usual DET v1 file; the frame
    indices must cover the shot exactly once, otherwise AV-POS-006 (new) and the run fails;
  - per frame results do not depend on neighbouring frames, so the merge is exact;
  - tracking then runs per shot as now. `track.overlap_pose` starts a shot's tracking when all its
    parts exist.
- **Batch grouping at chunk edges:** each process starts its batches of `pose.batch` frames at its
  first chunk, so the frames that share a batch differ from a single GPU run, wherever a chunk does
  not start at a multiple of the batch. Batching was equivalent within the noise floor (w10-b8),
  but it is named in the section 6 report as a known source of difference, next to the GPU
  numerics.
- **Failure:**
  - if one process fails, the run fails with that process's error, and the other is stopped;
  - an out of memory error halves the batch inside the process, as now;
  - there is no silent re-run on the other GPU, because that would change the numerics of the run.

## 4. Live mode keeps priority on GPU1

gpu_dual may start only if **all** of these hold at planning time:

- `.locks/gpu1.lock` is free (no live session, no other job);
- no live session is running or starting. LiveManager writes `.locks/live.session` (pid, start)
  while a session runs, and the planner also checks it;
- no compute process other than the resident llama-server (the VLM) runs on GPU1, and that process
  is not busy (utilisation under 10 % over 3 samples 1 s apart);
- GPU1 has enough free memory for the variant and batch (from the speed file, plus 1 GB of
  margin).

**If live starts during a dual job:**

- LiveManager tries to take gpu1.lock. Today this fails at once with AV-LIV-002; in the new design
  it posts a preempt request instead (`.locks/gpu1.preempt`);
- the GPU1 pose process checks for that request between batches. It finishes its current chunk,
  writes that chunk's part, records `preempted` (utc, the last chunk done), releases the lock and
  exits;
- the parent hands GPU1's remaining chunks to GPU0, in chunk order, after GPU0's own chunks;
- an active VLM request (llama-server on GPU1, which may stay resident) preempts the same way, at
  the next chunk boundary;
- live waits at most one chunk (at most `chunk_max_s` of video, roughly 10 to 15 s of pose work on the
  A4000). The page shows "GPU1 is finishing a batch chunk" with a countdown, then the session
  starts as today. `chunk_max_s` stays 30.

**Consequence:** a preempted run has chunks from both GPUs in a layout chosen by when live started,
so it is no longer bit reproducible. The plan records exactly which GPU did each chunk. It stays
valid because the A4000 against A5000 difference is checked against the noise floor in section 6.
If that check fails, a preempted run is marked "different" and must be repeated for the paper.

## 5. cpu_only

- It is used only when no CUDA device exists or when it is asked for, never because GPU0 is busy.
- After the first 50 frames the run measures CPU fps and logs
  `AV-RUN-008 (Notice): No GPU available; pose runs on the CPU at 2.1 fps, ETA 2 h 23 min`. The
  ETA is updated in the progress line, and the page shows the warning.
- Precision: FP32, batch 1, opencv reader (the reference profile without CUDA).
  `shots.parallel` and `overlap_stages` still apply.

## 6. Validation (public clips only, before the default changes)

All are run with `exp`, with stored configs and tables under experiments/:

1. **w10-gpu-a4000:** the default profile on GPU1 alone (`--gpu 1`, both public clips, 3 repeats),
   against w10-best-b8 on GPU0. This gives the per GPU numeric difference against the noise floor.
2. **w10-gpu-a5000:** today's reference. It is rerun the same day so the timing is comparable.
3. **w10-dual:** gpu_dual on both clips (3 repeats), against w10-gpu-a5000, for equivalence (noise
   floor rule), speed and the chunk layout. It is timed against the single GPU default with
   `shots.parallel` on, because dual mode turns parallel shots off. The report names the batch
   grouping at chunk edges (section 3) and the per GPU numerics as the known sources of difference.
4. **w10-dual-preempt:** a dual run with a scripted preempt after chunk 2. It checks the handover
   and the recorded plan.

Verdicts use the noise floor rule. If the A4000 alone is "different" from the A5000, gpu_dual stays
off for accuracy work, and the paper says so.

## 7. Requirement text drafts (for Khurshid)

**FR-INFRA-07 (replaces "each job is pinned to one GPU and the two are never combined"):**
Two GPUs, explicit assignment. A device planner assigns each job its device(s) before the first
stage and records the plan in the run manifest. A batch pose job may use both GPUs (gpu_dual), one
process per GPU on a fixed, deterministic chunk layout at shot cuts with shares from stored speeds,
only when GPU1 is idle and no live session runs. Its merged result must be equivalent within the
measured noise floor to a single GPU run. Every other job is pinned to one GPU. Live mode has
priority on GPU1: a dual job yields GPU1 at the next chunk boundary.

**NFR-ENV-08 (draft; the current wording is not in docs/, please check it replaces the right
sentence):** Every GPU process sees exactly one device through CUDA_DEVICE_ORDER=PCI_BUS_ID and
CUDA_VISIBLE_DEVICES, also in gpu_dual. Without a CUDA device the pipeline runs on the CPU (cpu_only)
with a warning and an estimated time. A busy GPU makes the job wait in the queue, never fall back to
the CPU silently. The compute plan, devices and locks of every run are stored in its manifest.

The HLD constraint in 2.3 ("the two are never combined for one job") and section 3.3.6 need the
same change. They also still name the GTX 1080 Ti, which has been replaced by the RTX A5000.

## 8. Decisions (Khurshid, 2026-11-12)

- **`"auto"`** stays opt in until section 6 passes and one QiSens run has been timed.
- **The live wait** of up to one chunk is fine, with the "GPU1 is finishing a batch chunk" message
  and a countdown; `chunk_max_s` stays 30.
- **The VLM** may stay resident on GPU1. An active VLM request preempts at the next chunk boundary,
  the same way live does.
