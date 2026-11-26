# Design changes since HLD v1.1 and LLD v1.0

**Status (2026-11-26).** A note for the next revision of the HLD (v1.1, Oct 21, 2026) and the LLD (v1.0). It lists every
design change found in the code and the experiment records, each with its reason and evidence. The HLD and LLD
themselves are not edited here. Evidence points to git commits on feat/w10-data, experiment folders under experiments/,
and the docs in docs_team/.

## Summary

| # | change | HLD / LLD place | kind |
| --- | --- | --- | --- |
| 1 | RTX A5000 instead of the GTX 1080 Ti | HLD 2.3, 3.3.6; CLAUDE.md rule 5 | hardware |
| 2 | Engine B zero shot is the default step engine | HLD 3.3.3 (Engine B, adoption rule) | method |
| 3 | The VLM scores the phases (it helps place the boundaries) | HLD 3.3.5, FR-VLM-03 | method, scope |
| 4 | Pose batch 8, TF32 and parallel stages | LLD pose and run manager sections | performance |
| 5 | FR-INFRA-07: one full run uses both GPUs | HLD 2.3, 3.3.6; LLD TC-RUN-03 | requirement changed |
| 6 | SQLite as an index over the run folders | HLD 3.3.6 storage; LLD 3.2 | implementation of the design |
| 7 | Login for the interface | LLD (open item "no login") | security |
| 8 | Audit log | LLD storage and interface | security, traceability |
| 9 | NFR-SEC-03: external access on the internal network | HLD 3.15, 3.3.4 | requirement changed |
| 10 | Report export per run, with revisions | HLD Visualization and Report module; LLD report builder | feature |
| 11 | 3D (GVHMR) and 4D (Pi3) as optional views | not in the HLD | new optional views |
| 12 | Improvement suggestions are rule based, no accept or reject flow yet | HLD 3.3.5, FR-MOTION-09 | method, scope |
| 13 | Added measures: cycle time graph, floor path in metres, tool pickup and object placement | HLD 3.3.7 Motion Metrics Design, Visualization and Report | feature |
| 14 | Validation freeze and the public train domain test | HLD 3.3.8 evaluation | process |

## 1. RTX A5000 instead of the GTX 1080 Ti

- **HLD said:** two unlike GPUs, a GTX 1080 Ti (11 GB, Pascal, FP32 only) for the pose pipeline and an RTX A4000 for the VLM
  (HLD 2.3, 3.3.6).
- **Now:** GPU0 is an RTX A5000 (24 GB, Ampere, bus 01:00) for the batch pose pipeline; GPU1 stays the RTX A4000 (16 GB)
  for the VLM and the live mode. torch 2.14.1+cu126 supports both Ampere cards.
- **Reason:** the server was upgraded (Week 9). The Ampere card also allows TF32 and FP16 and a larger pose batch.
- **Evidence:** commit 071ee68 (Week 9 hardware update, YOLO26 x default); experiments/w10_summary.md,
  w10-qisens-l-replica: the YOLO26 l run on the A5000 gives 75.02, identical to the 25 Oct run on the GTX 1080 Ti, so the
  new GPU did not change the result.

## 2. Engine B zero shot is the default step engine

- **HLD said:** Engine B is a learned temporal model (MS TCN or ASFormer style) pretrained on public assembly data and
  **fine tuned on the first 70 % of the QiSens footage**; it is used only if it beats Engine A on the held out last 30 %
  (F1@25 by at least 5 points). Otherwise Engine A stays the reported engine.
- **Now:** the default engine "B zero shot" combines the ordered dynamic programme decoder of Engine A with three zero shot
  phase likelihoods (SigLIP 2 full frame, SigLIP 2 subject crop, Qwen3 VL 8B; equal weights) and the Engine B v1 embedding
  change as boundary evidence. **No QiSens data is used for training or tuning.** Engine A and B v1 stay selectable.
- **Reason:** with one video, fine tuning on 70 % leaves a 3 minute test part with about two boundaries, too little for the
  adoption rule to mean anything, and every QiSens frame used for training is lost for testing. The learned Engine B v1 and
  v2 did not transfer from public data (67.26 and 52.50).
- **Evidence:** rule experiments/w11-zero-shot/rule.md; w11-qisens-zero-shot: Z1 (pre chosen) 83.14, Z2 (untuned equal
  weights) 93.20; w11-qisens-zero-shot-timing: full run 270.62 s, frame accuracy 92.97 to 93.40 over 3 repeats; phase
  descriptions frozen before the first QiSens run (commit 742bd1c). docs_team/validation_segmentation.md section 5 states
  that Z2 was made the default after both results were known.

## 3. The VLM scores the phases

- **HLD said:** the VLM output "is supplementary: it is never ground truth and never defines step boundaries"; it gives
  suggestions and optional step names from a closed vocabulary (HLD 3.3.5, FR-VLM-03).
- **Now:** in B zero shot the VLM gives a probability per phase for every 3 s window; these probabilities are one of the
  three likelihoods the decoder uses, so the VLM does influence where the boundaries fall. It is still never ground truth,
  and the decoder (not the VLM) places the boundaries under the phase order and duration limits.
- **Reason:** it is the largest single source of accuracy: without the VLM, Z2 falls from 93.20 to 66.21.
- **Evidence:** w11-qisens-zero-shot (ablation "Z2 with vlm off"); w11-vlm-speed (3 s windows, 2 frames, 4 slots). If the VLM
  server is not available, a run falls back to B v1 with notice AV-SEG-006, so the baseline results are still produced.

## 4. Pose batch 8, TF32 and parallel stages

- **LLD said:** one frame at a time through the pose model; stages one after the other.
- **Now:** pose batch 8 with TF32 (FP32 storage); shot detection runs in parallel with the pose pass (shots.parallel);
  pre and postprocessing run in threads around the pose inference (pose.overlap_stages); the VLM runs on GPU1 during the
  pose pass (zero_shot.vlm_parallel).
- **Reason:** speed. The full QiSens run had to stay under 300 s with the VLM added.
- **Evidence:** experiments/INDEX.md, default profile of 2026-11-08 (w10-best-b8); w10-shots-par and w10-overlap-stages
  (equivalent results); the equivalence rule and noise floor of w10-ref-notf32. Engine A steps, NULL decisions and flags
  were identical; the metric changes above the noise floor are listed in w10_summary.md. Accuracy checks use
  configs/reference.toml (batch 1, strict FP32).

## 5. FR-INFRA-07: one full run uses both GPUs

- **HLD said:** "Each job is pinned to one GPU and the two are never combined for one job" (HLD 2.3); LLD TC-RUN-03: a GPU
  worker sees exactly one device.
- **Now:** a full run with B zero shot uses GPU0 for pose (and SigLIP, body3d, scene4d) and GPU1 for the VLM at the same
  time. Each stage still sees one device and takes its GPU's lock (.locks/gpu<N>.lock); no single model is split over two
  GPUs. Proposed new wording: "Each stage is pinned to one GPU; one run may use both GPUs for different stages, each under
  its GPU lock."
- **Reason:** the VLM needs its own GPU, and running it in parallel keeps the full run under 300 s.
- **Evidence:** w11-qisens-zero-shot-timing (pose on GPU0, Qwen3 VL 8B on GPU1, 270.62 s); notice AV-SEG-006 when GPU1 is busy.

## 6. SQLite as an index over the run folders

- **HLD and LLD said:** a SQLite database (runs, shots, track index, step segments, metrics, suggestion log, evaluations)
  plus a file store; the database references files by path.
- **Now:** implemented in the week of 2026-11-14 as one database per data area (data/qisens/av.db, data/public/av_dev.db,
  so QiSens rows never leave data/qisens), with tables runs, steps, metrics, evaluations, notices and revisions. The run
  folders (manifest.json, steps.csv, metrics.csv, eval.json) stay the record the pipeline writes; the database indexes them
  and is rebuilt from them when files change. There is no suggestion log table yet (see 12).
- **Reason:** the pipeline was built file first; making the files the source of truth keeps every run reproducible and
  lets the database be rebuilt at any time.
- **Evidence:** commit 6c4b13a (Item 3c: SQLite Repository behind RunStore, with a migration of the existing runs);
  storage/repository.py; tests test_repository.

## 7. Login for the interface

- **LLD said:** "No login: access control is the SSH account on the server (open item)."
- **Now:** login with a user name and password (salted hashes, file mode 600); 5 wrong passwords lock the name for 5
  minutes; served media paths are checked to lie inside an open data area.
- **Reason:** the TA review asked for access control, and the interface became reachable on the internal network (9).
- **Evidence:** commits 921a8e4 (Item 3d: login, safe media paths, audit log), c7137a9, e122c15 (lockout); tests test_ui_auth.

## 8. Audit log

- **Now:** two append only logs. Every login (with the reason of a failure) and every lockout goes to logs/ui_audit.jsonl
  (user name, result, reason; no QiSens content). Every run start, report export and approved correction goes to
  audit.jsonl inside the data area of the run (data/qisens/audit.jsonl for QiSens runs). No password is ever written to
  either.
- **Reason:** traceability of who did what on QiSens data.
- **Evidence:** commit 921a8e4; ui/auth.py; test_ui_auth (no password in the log).

## 9. NFR-SEC-03: external access on the internal network

- **HLD said:** "The application binds locally and is reached through an SSH port forward or the internal network. No share link."
- **Now:** the host is a setting (ui.host). The default stays 127.0.0.1; the team runs it on 0.0.0.0 for the internal
  network, which the interface allows only with login users set up. A public share link is still never used, and the
  QiSens data still never leaves the server.
- **Reason:** the mentor's decision of 2026-11-15, so that the mentor and the professor can open the results directly.
- **Evidence:** commit e122c15 (interface host as a setting, the mentor's decision); ui/app.py check_bind (AV-APP-005 for a
  public address, AV-APP-009 without login users).

## 10. Report export per run, with revisions

- **HLD said:** a PDF baseline report (ReportLab), downloadable from the interface.
- **Now:** per run export to export/rev<N>/ with report.pdf (Matplotlib PDF), report.json, steps.csv and metrics.csv;
  revision 0 is the run as computed. A boundary correction in the interface makes a new revision with recomputed metrics,
  and never changes the run files.
- **Reason:** the TA review (Item 3a) and the need to correct boundaries without losing the original result.
- **Evidence:** commit d178469 (Item 3a: report export); output/report.py, output/corrections.py; tests test_report_export
  and test_corrections.

## 11. 3D and 4D as optional views

- **HLD:** no 3D or 4D view.
- **Now:** two optional stages, off by default: body3d (GVHMR, the worker's skeleton in metres on a floor plane) and
  scene4d (Pi3, the scene as a point cloud with the person, the camera path and the aligned skeleton). They are views
  in the Results page; no step or metric depends on them. The GVHMR floor path in metres is used by measure 2 (see 13).
- **Reason:** a 2D image measure depends on the camera view; a metric floor map and a 3D view help the mentor read the
  motion. Kept optional because of the GPU time.
- **Evidence:** commits d71b2c0, 75bb5ba (body3d); rule and result experiments/w11-scene4d-model (Pi3 chosen),
  w11-scene4d-quality, w11-scene4d-fusion (fusion did not win; Pi3 kept).

## 12. Improvement suggestions are rule based

- **HLD said:** a limited set of labelled VLM suggestions on sampled frames, each logged with the user's accept or reject
  (FR-MOTION-09, HLD 3.3.5).
- **Now:** the suggestions in the Results page and the PDF come from a rule list on the measured metrics (bending, idle,
  two hand use, repetition, long phase, hand travel, walking), each with its evidence (metric, value, threshold, time
  ranges), labelled as suggestions and not findings, and marked experimental. **There is no accept or reject flow and no
  suggestion log yet**, and the VLM writes no suggestions.
- **Reason:** a rule based suggestion can show its evidence and cannot invent content; the VLM's time went to the phase
  scores (3).
- **Evidence:** output/extras.py; docs_team/validation_measurements.md (measure 5, target: at least 60 % rated useful).

## 13. Added measures

- cycle time graph (one bar per cycle when a phase repeats), in Results and the PDF;
- movement path in metres on the GVHMR floor, per phase (measure 2, not validated);
- tool pickup and object placement (measures 7 and 8, experimental): a YOLO26 m COCO object detector on the CPU and the
  wrist speed; 0 events on 3 public runs, because COCO has almost no workshop tools.
- **Evidence:** docs_team/validation_measurements.md and demo_outputs.md; the report is outside the frozen method
  (Khurshid's decision of 2026-11-26) and these measures do not change steps.csv or metrics.csv.

## 14. Validation freeze and the public train domain test

- **HLD said:** evaluation on the held out last 30 % of the QiSens video and on public benchmarks.
- **Now:** git tag validation-freeze-1 fixes the code, configs, phase descriptions and the model checksums; every
  validation result records it. The single video is reported as a development result, with a robustness study
  (experiments/w12-robustness) and a public train domain test (14 clips: 13 correct, 92.86 %; joined video 89.69).
- **Reason:** Professor Kim's review: with one QiSens video the evidence has to come from robustness, an independent
  public domain and honest reporting.
- **Evidence:** docs_team/freeze/validation-freeze-1.json; docs_team/validation_segmentation.md; docs_team/train_domain_test.md.
