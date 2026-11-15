# Engine B: setup and use in your own account (guide for Dr. Choi)

Written 2026-11-15 for the copy in the qisens1 account. Every command was tested in a fresh copy in a
temporary folder of the qisens account (section 5). Nothing here reads or changes anything outside the
two AssemblyVision folders.

## 1. What Engine B is now

Engine B is the step engine "B" of the interface (the preset "B zero shot" in code; the default). It
finds the five ordered process phases (Removal, Disassembly, Cleaning, Adjustment and Inspection,
Assembly and Installation) **zero shot**: no QiSens label was used to train or tune it.

- **Phase scores by meaning.** Each phase has a frozen text description (docs_team/phase_descriptions.md,
  written before any QiSens run).
  - **SigLIP 2** (so400m, 384 px) compares 1 frame per second, as the full frame and as a crop of
    the worker, with the descriptions of each phase.
  - **Qwen3 VL 8B** (4 bit GGUF in llama.cpp) looks at a short clip every 3 s (2 frames) and returns a
    probability per phase.
- **Pose evidence.** YOLO26 x pose with ByteTrack gives the worker's keypoints. The Engine B v1 model
  turns them into a motion embedding whose changes mark likely boundaries.
- **Decoder.** A dynamic programme puts the K - 1 boundaries in the known phase order. It maximises the
  boundary evidence plus the phase log likelihood of the zero shot scores, with equal weights (1, 1, 1),
  untuned.
- **Fallback.** If GPU 1 is busy or the VLM fails, the run uses the pose evidence alone (step engine
  "B v1") and says so (notice AV-SEG-006).

**Results on the QiSens video** (9.9 min, blind, 3 full runs):

| measure | value |
| --- | --- |
| frame accuracy | 92.97 to 93.40 (median 93.40) |
| edit score | 100 |
| F1@10, F1@25, F1@50 | 100, 100, 100 |
| boundary MAE | 7.85 to 8.37 s |
| full run time (RTX A5000 and A4000) | 270.6 s |

The rules alone (Engine A) give 50.00 and the pose evidence alone gives 67.26 (experiments/w10_summary.md).
The runs differ a little because the VLM's parallel requests are batched differently each time
(experiments/w11-vlm-determinism); `--set zero_shot.vlm_slots=1` makes it bit identical, at about 1.7x
the VLM time.

## 2. Copy from the qisens account (sudo)

The copy takes the code of the `main` branch and the model files from /home/qisens/AssemblyVision. It
leaves out data/ (both data areas), .claude-home/ and every local folder: envs, caches, logs, locks,
the interface users (.config) and the llama.cpp build. Your own data/, .config/ and envs are never
touched; files of the code with the same name are replaced, so save local code changes first.

```bash
SRC=/home/qisens/AssemblyVision
DEST=/home/qisens1/AssemblyVision        # your copy; av.sh expects ~/AssemblyVision

# 0. The source must be on main with no local changes (both lines must print the same commit, and nothing else)
sudo -u qisens git -C $SRC rev-parse main HEAD
sudo -u qisens git -C $SRC status --porcelain --untracked-files=no

# 1. Code (main) and the repository history
sudo rsync -a \
  --exclude=/data/ --exclude=/.claude-home/ --exclude=/env/ --exclude=/envs/ \
  --exclude=/tmp/ --exclude=/logs/ --exclude=/.locks/ --exclude=/.cache/ --exclude=/.config/ \
  --exclude=/tools/ --exclude=/models/ --exclude=/third_party/ --exclude='*.bak' \
  $SRC/ $DEST/

# 2. The model files (the list in scripts/engine_b_models.sha256, 11 GB)
sudo rsync -a --files-from=<(awk '{print $2}' $SRC/scripts/engine_b_models.sha256) $SRC/ $DEST/

# 3. llama.cpp source at tag b12005, without its build (it is built in your copy, section 3)
sudo mkdir -p $DEST/third_party
sudo rsync -a --exclude=/build/ $SRC/third_party/llama.cpp/ $DEST/third_party/llama.cpp/

# 4. Ownership, branch, checksums
sudo chown -R qisens1:qisens1 $DEST
cd $DEST && git checkout main && sha256sum -c scripts/engine_b_models.sha256
```

Every line of the check must end in `OK`. The model files and their SHA256:

| file | size | SHA256 | source |
| --- | --- | --- | --- |
| models/yolo26x-pose.pt | 126 MB | 08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d | Ultralytics assets v8.4.0 (AGPL 3.0) |
| models/yolo26n-pose.pt, s, m, l | 8 to 58 MB | in scripts/engine_b_models.sha256 | Ultralytics assets v8.4.0 (smaller pose models for the interface) |
| models/engine_b/w10-engine-b-v1.pt | 1.4 MB | 9e8f18f2103bd17be4d23701d627d809059ba4da06bc7ebebb47d46940573fe9 | ours (trained on HA4M, public); not downloadable, only copied |
| models/siglip2/so400m-patch14-384/model.safetensors | 4.5 GB | 9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2 | Hugging Face google/siglip2-so400m-patch14-384, revision e8e487298228002f3d8a82e0cd5c8ea9c567f57f (Apache 2.0); 6 small config and tokenizer files with it, in the list |
| models/Qwen3VL-8B-Instruct-Q4_K_M.gguf | 5.0 GB | 67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2 | Hugging Face Qwen/Qwen3-VL-8B-Instruct-GGUF (Apache 2.0) |
| models/mmproj-Qwen3VL-8B-Instruct-F16.gguf | 1.2 GB | ca524100ebf825c9a870db1c580d03879e0da0ab2541697e2458e64891cf9d38 | the same repository |

The Engine B v1 model alone, if you only need that one file:

```bash
sudo cp /home/qisens/AssemblyVision/models/engine_b/w10-engine-b-v1.pt $DEST/models/engine_b/
sudo chown qisens1:qisens1 $DEST/models/engine_b/w10-engine-b-v1.pt
echo "9e8f18f2103bd17be4d23701d627d809059ba4da06bc7ebebb47d46940573fe9  $DEST/models/engine_b/w10-engine-b-v1.pt" | sha256sum -c
```

## 3. Environments and llama.cpp (as qisens1, no sudo)

Conda envs hold absolute paths, so ours cannot be copied. Both envs are made again from exact specs in
scripts/ (env.conda-spec.txt and llamabuild.conda-spec.txt for conda, code/requirements-pinned.txt for
pip, torch 2.14.1+cu126). This was chosen over conda-pack, which is not installed and would have to be
added to the shared anaconda. About 5 GB are downloaded; every cache stays in your copy.

```bash
cd ~/AssemblyVision
[ -d env ] && mv env env.old                  # an older env of your copy, if there is one
scripts/setup_env.sh                          # env/ and envs/llamabuild/, about 5 min
scripts/build_llama.sh                        # third_party/llama.cpp/build/bin/llama-server, about 12 min
source ~/AssemblyVision/av.sh                 # activates env/, sets the caches; every command below assumes it
cd code && python -m unittest discover -s tests     # CPU tests, about 70 s, must end in OK
```

llama-server is built in each copy because it finds its libraries by an absolute path into the copy's
envs/llamabuild (an RPATH); a copied binary would look into the qisens folder.

## 4. Commands

Run from ~/AssemblyVision/code after `source ~/AssemblyVision/av.sh`. QiSens videos and everything made
from them stay under data/qisens/ of your copy; public videos under data/public/.

**A full run with Engine B** (YOLO26 x on GPU 0, the VLM on GPU 1), with the evaluation when a GT file is given:

```bash
python -m assemblyvision.tools.run_pose --source qisens --video ../data/qisens/videos/VIDEO.mp4 \
  --variant x --gpu 0 --gt ../data/qisens/annotations/steps_gt.csv
```

The run folder data/qisens/runs/run_<date>_<time>_<id>/ holds steps.csv, metrics.csv, eval.json,
siglip.npz, vlm.npz (scores only), the overlay video and manifest.json (every stage, time and notice).

**Evaluation of a finished run against a GT file** (CPU; the GT CSV has step,label,start_s,end_s,source):

```bash
python -m assemblyvision.tools.run_pose --run ../data/qisens/runs/RUN_ID --stages eval --gt ../data/qisens/annotations/steps_gt.csv
python -m assemblyvision.tools.run_evals --source qisens --runs RUN_ID [RUN_ID ...]     # several runs, min, median, max
```

**Compare mode** (CPU, in memory, writes nothing): segment the stored run once per value of one
setting and score each against the GT, for example Engine B with and without the VLM, or Engine A:

```bash
python -m assemblyvision.tools.run_pose --run ../data/qisens/runs/RUN_ID --gt ../data/qisens/annotations/steps_gt.csv \
  --set 'seg_a.engine="custom"' --set 'engine_b.model="engine_b/w10-engine-b-v1.pt"' --set 'engine_b.candidates="dense"' \
  --set seg_a.embed_weight=0.2 --set seg_a.embed_window_s=30.0 --set seg_a.siglip_weight=1.0 --set seg_a.siglip_crop_weight=1.0 \
  --compare seg_a.vlm_weight=1.0,0.0
python -m assemblyvision.tools.run_pose --run ../data/qisens/runs/RUN_ID --gt ../data/qisens/annotations/steps_gt.csv \
  --compare seg_a.engine='"B zero shot","B v1","A"'
```

The second form compares step engines: B, B with the pose evidence alone (B v1) and the rules (A).

**Export** (CPU): PDF, JSON and CSV with timestamps, phase confidence, provenance (commit, models with
SHA256, config, GPU) and notices, into the run's export/rev0/:

```bash
python -m assemblyvision.tools.export_run --run ../data/qisens/runs/RUN_ID
```

**The interface** (login required; ports 7860 to 7869):

```bash
python -m assemblyvision.tools.ui_users add choi          # once per user, asks for the password twice
python -m assemblyvision.tools.ui_users passwd choi       # a new password later (works at the next login)
python -m assemblyvision.ui.app --sources qisens,public --port 7861
# from your laptop: ssh -L 7861:127.0.0.1:7861 qisens1@<server>, then open http://127.0.0.1:7861
```

**Host and port.** By default the page listens on 127.0.0.1 (this server only; reach it through the
ssh tunnel above). To reach it from other computers, listen on every interface (0.0.0.0) or on one
private address of the server, either in the configuration or on the command line (the command line wins):

```toml
# code/configs/default.toml, section [ui] (or the same lines in a file given with --config)
host = "0.0.0.0"
port = 7861
```

```bash
python -m assemblyvision.ui.app --sources qisens,public --host 0.0.0.0 --port 7861
# then open http://<server address>:7861 from another computer
```

Any host other than 127.0.0.1 needs at least one login user; without one the page does not start
(AV-APP-009), and --no-login works only on 127.0.0.1. Five wrong passwords lock a user name for
5 minutes (logs/ui_audit.jsonl records every login, failure and lockout, never a password). A public
address of its own and share links stay refused.

**If the page cannot be reached from outside:**

1. It listens where you expect: `ss -tlnp | grep 7861` must show `0.0.0.0:7861` (or the private
   address), not `127.0.0.1:7861`.
2. It answers on the server: `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7861/` prints 401
   (the login page) or 200.
3. The firewall lets the port in: `sudo ufw status`; if ufw is active, `sudo ufw allow 7861/tcp`
   (better: only from your network, `sudo ufw allow from 192.168.0.0/24 to any port 7861 proto tcp`).
4. From outside the building network, the router must forward the port to the server (port forwarding
   in the router's settings), or use the ssh tunnel instead.
5. The address is http, not https (the page has no certificate): type `http://` explicitly; a browser
   that forces https cannot connect.
6. The Live page uses the camera of the browser, and browsers allow the camera only on https or on
   localhost. Over http://<server address> the camera is refused; use the ssh tunnel
   (http://127.0.0.1:7861) for Live mode.

In the interface: New analysis, choose the video, pose model x and step engine B. Results shows the
phases, the metrics, the boundary correction (saved as a new revision) and the report export.

**GPUs and locks.** GPU 0 (RTX A5000) runs pose and SigLIP, GPU 1 (RTX A4000) runs the VLM; each job holds
.locks/gpu<N>.lock of its own copy, so jobs of one copy wait for each other. The locks of the two
accounts are separate: agree with the qisens account before starting a run, or the two runs share the GPUs.

## 5. How this guide was tested

On 2026-11-15, in /home/qisens/AssemblyVision/tmp/choi_test/AssemblyVision (deleted afterwards), as the qisens
user, so without sudo and chown:

| step | result |
| --- | --- |
| copy (section 2) | code 10 s, models 11 GB; `sha256sum -c` all OK |
| scripts/setup_env.sh | 5 min; torch 2.14.1+cu126 with CUDA |
| scripts/build_llama.sh | 12 min; llama-server b12005, every library from the copy's envs/llamabuild |
| CPU tests | 488 tests OK (2 skipped) |
| full run with Engine B on a public IKEA ASM video (40 s, Kallax, its own step list) | 48 s; VLM and SigLIP stages done on GPU 1 and 0; frame accuracy 57.21 against its GT |
| evaluation, run_evals, both compare forms, export | as in section 4 (rev0 with PDF, JSON, CSV) |
| interface | 401 before the login, 200 after it |

The public video needs `--config` with its own step list and `--set 'zero_shot.phases="ikea_asm:Kallax_Shelf_Drawer"'`;
QiSens videos use the defaults.
