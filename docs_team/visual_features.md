# Visual scene features as Engine B evidence (design, 2026-11-12)

Status: **design for Khurshid's review.** Nothing is installed, downloaded or run until he
approves. No GPU job without asking.

## 1. Why

- Engine B v1 and v2 only see the worker's pose. v2 reached 88.53 on the HA4M macro phase
  benchmark but 52.50 on QiSens blind (w10-qisens-engine-b-v2): a pose embedding trained on HA4M
  phases does not carry over to another task.
- A phase change on the QiSens line usually also changes the scene: other parts on the table, a
  different tool, a different place on the bogie, often a new shot.
- A frozen, general purpose image encoder sees this without being trained on any phase labels, so
  there is nothing task specific to overfit. That is the lesson of v2.

## 2. Encoder

Recommendation: **DINOv2 ViT S/14**.

| | DINOv2 ViT S/14 (recommended) | CLIP ViT B/32 (alternative) |
| --- | --- | --- |
| Source | Meta, `facebookresearch/dinov2` | OpenAI weights through `open_clip` |
| Licence | Apache 2.0, code and weights (official README: "DINOv2 code and model weights are released under the Apache License 2.0") | MIT (code and weights). The OpenAI model card puts deployed use out of scope; a research benchmark is fine, the product would need a check. |
| Size | 21 M parameters, weights about 85 MB (`dinov2_vits14_pretrain.pth`) | about 88 M parameters in the image tower; the checkpoint with the text tower is about 600 MB |
| Embedding | 384 (CLS token), unit length after normalisation | 512 |
| Training | self supervised (no labels, no text): good at scene layout and objects | image and text pairs: good at semantic categories, weaker at layout |
| Install inside env | **no pip package.** `torch.hub` code plus one weight file under ~/AssemblyVision/models/ (commands below) | `pip install open_clip_torch==<pinned>` into env (pulls `timm`, `ftfy`, `regex`); weights under models/ |
| GPU memory (estimate; measured in the pilot) | under 1 GB at batch 64, 224 px, FP32 | about 1.5 GB at batch 64, 224 px, FP32 |
| Speed (estimate) | several hundred images per second on the A5000; at 2 fps a 10 min video is 1200 images, a few seconds of compute | about the same |

**Why DINOv2:**
- no new pip packages (it runs on the torch 2.14.1 we have, so there is no risk to the env);
- the clearest licence;
- it is the smallest;
- its self supervised features follow layout and objects, which is what changes between phases.

**Install** (only after approval; everything stays inside ~/AssemblyVision):

```
source ~/AssemblyVision/av.sh
export TORCH_HOME=~/AssemblyVision/models/torch_hub
git clone https://github.com/facebookresearch/dinov2 ~/AssemblyVision/models/torch_hub/dinov2   # then git checkout <pinned commit>
curl -L -o ~/AssemblyVision/models/dinov2/dinov2_vits14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
sha256sum ~/AssemblyVision/models/dinov2/dinov2_vits14_pretrain.pth >> ~/AssemblyVision/models/SHA256SUMS_dinov2.txt
```

- At run time the model is loaded with
  `torch.hub.load(<local clone>, "dinov2_vits14", source="local", pretrained=False)`,
  and the weight file is loaded after its checksum is checked.
- There is no network access at run time, and nothing leaves the server.
- The encoder code is a pinned clone. It is never updated by hub.

## 3. Pipeline

- **New stage `visual`**, after pose, in the same run_pose process (GPU0, under the same
  .locks/gpu0.lock, so it is never a second job on the GPU).
  - Frames are sampled at `visual.fps` (2) from the CFR video with the existing decoder.
  - They are resized (short side 224, centre crop 224) and normalised (ImageNet mean and std).
  - Then batched (`visual.batch` 64) and embedded.
- **Output:** `visual.npz` in the run folder (time_s [N], embedding [N, 384] float16, model and
  checksum). For a QiSens run that is under data/qisens/ like every other run file.
  - Frames are never written to disk.
  - QiSens frames only ever go through this local model.
- **Rerun and compare:** the stage is stored, so `run_pose --compare` and the benchmark can reuse
  the embeddings on the CPU without the GPU.
- **Decoder (Engine B):**
  - the visual change score is boundary_tcn.change_score on the 2 fps embeddings (window
    `seg_a.visual_window_s` before and after t), divided by its maximum over the active span,
    then sampled at the dense candidate grid;
  - a dense candidate gets `strength += seg_a.visual_weight * visual change`;
  - Engine A candidates get the same term at their frame;
  - the DP and its limits stay as they are;
  - the edge guard of the dense candidates applies (half a window from the ends).
- **Defaults:** `visual.enabled = false`, `seg_a.visual_weight = 0`, so nothing changes until a
  preset is chosen by the rule.

New config keys, all off or inert by default:

```
[visual]
enabled = false
model = "dinov2_vits14"
fps = 2.0
input_px = 224
batch = 64
half = false
[seg_a]
visual_weight = 0.0
visual_window_s = 20.0
```

## 4. Selection set (public RGB with long procedural phases)

HA4M RGB is too slow to download (about 45 min per video). The proposals:

| dataset | why | phases | licence | download |
| --- | --- | --- | --- | --- |
| **50 Salads** (Stein and McKenna 2013) | fixed top view of one worker at a table; 50 videos of about 6 min; the same procedure in a varying order | 3 high level activities (cut and mix ingredients, prepare dressing, serve salad), tens of seconds to minutes, given in the official annotations | `CC BY-NC-SA 4.0` (official Dundee page) | official page: RGB videos about 1.8 GB, annotations 5 MB |
| **IKEA ASM** (Ben-Shabat et al. 2021) | furniture assembly, closest to our task; 371 assemblies, 3 RGB views | 33 atomic actions, no phase labels: macro phases would be grouped by hand (like HA4M) and written before any run | `CC BY-NC 4.0` (project page; code MIT) | Google Drive zips; total size not stated (to check before download) |
| **HA ViD** (Zheng et al. 2023) | human assembly of a generic assembly box, 3 Azure Kinect views, 3222 videos | HR SAT action labels; subassembly level grouping to be defined | `CC BY-NC 4.0` (project page) | request form and code of conduct; size not stated |

**Proposal:**
- **50 Salads first.** Its phases come from the dataset itself, not from our grouping, which avoids
  the HA4M trap of judging a model on a grouping we made. It is small, and its single fixed view
  is close to the QiSens fixed shots.
- **Then IKEA ASM** (one view, front) as the second, assembly domain set, if its size is
  reasonable.
- The selection score is the worst case over the two datasets, so no single dataset decides.

**Licences:** all three are non commercial (`NC`). That fits the capstone research and the paper.
Any commercial use by QiSens would need the authors' permission. Khurshid decides whether NC is
acceptable.

Every download goes to data/public/datasets/<name>/ with SOURCE.json (URL, licence, citation)
and SHA256SUMS, as for HA4M. A download over 5 GB asks first.

## 5. Selection rule (draft; it is committed before any run)

- **Splits:**
  - 50 Salads: the 5 official folds by subject.
  - IKEA ASM: its official train and test split.
  - Settings are chosen by CV on the train parts only. The test parts are scored once.
- **Grid:**
  - `seg_a.visual_weight` 0, 0.5, 1, 2;
  - `seg_a.visual_window_s` 10, 20, 40;
  - with and without the B v1 pose term (B v1 settings);
  - `seg_a.engine_a_scale` 1 and 0.5.
  - `visual.fps` stays 2 (a speed choice, not an accuracy one).
- **Criterion:** the worst case over the datasets of the CV mean frame accuracy. Ties go to the
  smaller visual weight.
- **Disclosed beforehand:** the QiSens results known so far (B v1 67.26, v2 52.50, Engine A
  50.00). The QiSens video is never used to choose.
- **QiSens:** one blind compare of the chosen setting against B v1. The pre chosen headline is
  the chosen setting.

## 6. Risks

- **The view changes the embedding more than the work does.** QiSens has handheld shots: a cut
  or a camera move changes the scene embedding without a phase change. Mitigations:
  - the change score uses windows of tens of seconds;
  - the cut candidates already mark shot changes;
  - an option to compute the change only within a shot is part of the grid if the pilot shows
    the problem.
- **Lighting and people walking through:** these are window averages over 10 to 40 s, so short
  events wash out.
- **Speed:** the new stage adds the decode of 2 fps frames plus a few seconds of GPU per 10 min
  video. It is timed in the pilot against the speed gate.

## 7. Steps after approval

1. Install the encoder (section 2), checksum, and a unit test with a fixed image (embedding
   shape, determinism).
2. The `visual` stage, the decoder term, tests on synthetic videos (a scene change gives a
   boundary).
3. Download 50 Salads (about 1.8 GB, under the 5 GB limit) and the IKEA ASM size check.
4. Commit the rule.
5. GPU pilot on 2 public videos (asked first): memory and time.
6. Embed the selection set (GPU, asked first), CV grid on CPU, test once.
7. Write the QiSens command for Khurshid.
