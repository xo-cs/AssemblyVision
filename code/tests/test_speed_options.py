"""Week 10 speed options with a fake model on the CPU: batched pose, prefetch
reader, frames by seeking (shots.motion_seek), reference frames from the pose
pass, tracking in a worker process and the chunked overlay. Each must give the same files as the reference path."""
import hashlib
import io
import json
import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from assemblyvision.config import Config
from assemblyvision.errors import AvError
from assemblyvision.ingest.ingestor import frame_reader, iter_frames, iter_frames_at, iter_frames_prefetch
from assemblyvision.ingest.shots import ShotDetector
from assemblyvision.output import overlay
from assemblyvision.perception.pose import PoseEstimator, load_det
from assemblyvision.perception.tracker import Tracker
from assemblyvision.tools.run_pose import _track_worker_init, track_shot_job
from assemblyvision.types import Shot
from helpers import fake_root, ffmpeg_clip, make_ctx
from motion_helpers import pose_series


def no_seconds(x):
    """The records without their timing fields (they differ from run to run)."""
    if isinstance(x, dict):
        return {k: no_seconds(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [no_seconds(v) for v in x]
    return x


class _Boxes(SimpleNamespace):
    def __len__(self):
        return self.xyxy.shape[0]


def frame_result(frame: np.ndarray, conf_min: float):
    """Deterministic from the pixels: up to three people, confidences 0.9, 0.6, 0.3
    (the strong ones only from the 6th frame on, read from the frame's brightness)."""
    v = float(frame[::8, ::8].mean())
    k = int(frame[::16, ::16, 1].sum()) % 7
    confs = np.array([0.3, 0.6, 0.9] if k >= 2 else [0.3], np.float32)
    keep = confs >= conf_min
    n = int(keep.sum())
    base = torch.tensor([[10.0 + 40 * i + v % 7, 20.0 + k, 60.0 + 40 * i, 200.0] for i in range(len(confs))])
    kp = torch.stack([torch.full((17, 3), 0.5) + 0.01 * i for i in range(len(confs))])
    kp[:, :, 0] += v
    return SimpleNamespace(boxes=_Boxes(xyxy=base[keep].reshape(n, 4), conf=torch.from_numpy(confs[keep])),
                           keypoints=SimpleNamespace(data=kp[keep]),
                           speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 1.0})


class FakeModel:
    def __init__(self, fail_when=None, oom_above=None):
        self.calls = []
        self.fail_when, self.oom_above = fail_when, oom_above

    def predict(self, source, conf=0.1, **kw):
        frames = source if isinstance(source, list) else [source]
        self.calls.append(len(frames))
        if self.oom_above is not None and len(frames) > self.oom_above:
            raise torch.cuda.OutOfMemoryError("fake")
        if self.fail_when is not None and any(self.fail_when(f) for f in frames):
            raise RuntimeError("bad frame")
        return [frame_result(f, conf) for f in frames]


class SpeedOptionsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        cfg, ctx = make_ctx(self.root)
        videos = ctx.paths.root.parent.parent / "videos"
        videos.mkdir()
        self.video = ffmpeg_clip(videos / "c.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", seconds=2)
        self.shots = [Shot("s01", "r", 0, 20, 0.0, 0.8, "handheld"), Shot("s02", "r", 20, 50, 0.8, 2.0, "handheld")]

    def tearDown(self):
        self._tmp.cleanup()

    def pose(self, model=None, **pose_cfg):
        cfg = Config.load(overrides={"pose": pose_cfg}) if pose_cfg else Config.load()
        cfg_, ctx = make_ctx(self.root)
        p = PoseEstimator(cfg, ctx, device="cpu", models_dir=self.root / "models")
        p.model, p.variant = model or FakeModel(), "x"
        return p, ctx

    def dets(self, p, ctx, **kw) -> dict:
        paths = p.estimate_body_shots(ctx.run_id, self.shots, video=self.video, **kw)
        return {sid: load_det(path) for sid, path in paths.items()}

    def assertSameDets(self, a: dict, b: dict):
        self.assertEqual(set(a), set(b))
        for sid in a:
            for key in ("frame_idx", "bbox_xyxy", "det_conf", "body_xy", "body_conf"):
                np.testing.assert_array_equal(a[sid][key], b[sid][key], err_msg=f"{sid} {key}")

    def test_batched_pose_gives_the_same_det_files(self):
        p1, c1 = self.pose()
        ref = self.dets(p1, c1)
        for batch in (8, 16, 32):
            pb, cb = self.pose(batch=batch)
            self.assertSameDets(ref, self.dets(pb, cb))
            self.assertEqual(max(pb.model.calls), batch)
            m = cb.manifest["pose_body"]
            self.assertEqual((m["batch"], m["batch_used"], m["frames"]), (batch, batch, 50))
            bd = m["breakdown_s"]
            self.assertEqual(set(bd), {"decode", "infer_call", "preprocess", "inference", "postprocess", "extract",
                                       "writing"})
            self.assertAlmostEqual(bd["inference"], 50 * 0.005, places=6)    # from result.speed

    def test_failed_batch_retries_frame_by_frame(self):
        bad = {"seen": None}

        def fail_when(frame):            # one particular frame always fails
            h = hashlib.md5(frame.tobytes()).hexdigest()
            bad["seen"] = bad["seen"] or h
            return h == bad["seen"]
        p, ctx = self.pose(FakeModel(fail_when=fail_when), batch=8)
        self.dets(p, ctx)
        m = ctx.manifest["pose_body"]
        self.assertEqual((m["failed_frames"], m["frames"]), (1, 50))
        self.assertEqual(m["per_shot"][0]["failed_frames"], 1)

    def test_out_of_memory_halves_the_batch_first(self):
        p, ctx = self.pose(FakeModel(oom_above=4), batch=16)
        p1, c1 = self.pose()
        self.assertSameDets(self.dets(p1, c1), self.dets(p, ctx))
        m = ctx.manifest["pose_body"]
        self.assertEqual((m["batch_used"], m["fallback"]), (4, None))

    def test_prefetch_and_reader_choice_give_the_same_frames(self):
        want = (lambda i: i % 3 != 1)
        a = [(i, hashlib.md5(f.tobytes()).hexdigest()) for i, f in iter_frames(self.video, 5, 40, want)]
        b = [(i, hashlib.md5(f.tobytes()).hexdigest()) for i, f in iter_frames_prefetch(self.video, 5, 40, want, 3)]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)          # 35 frames, 11 of them i % 3 == 1
        c = [(i, hashlib.md5(f.tobytes()).hexdigest()) for i, f in frame_reader("prefetch", 2)(self.video, 5, 40, want)]
        self.assertEqual(a, c)
        p1, c1 = self.pose()
        pp, cp = self.pose(decode="prefetch", batch=8)
        self.assertSameDets(self.dets(p1, c1), self.dets(pp, cp))
        with self.assertRaises(AvError):
            list(iter_frames_prefetch(self.root / "missing.mp4"))

    def test_seek_gives_the_same_frames(self):
        a = [hashlib.md5(f.tobytes()).hexdigest() for _, f in iter_frames(self.video, 17, 30)]
        b = [hashlib.md5(f.tobytes()).hexdigest() for _, f in iter_frames(self.video, 17, 30, seek=True)]
        self.assertEqual(a, b)

    def test_frames_at_indices_match_a_full_decode(self):
        want = [3, 4, 18, 19, 47, 48, 49]
        a = [(i, hashlib.md5(f.tobytes()).hexdigest()) for i, f in iter_frames(self.video, want=set(want).__contains__)]
        for gap in (0, 5, 60):             # 0 seeks on every jump, 60 grabs through
            b = [(i, hashlib.md5(f.tobytes()).hexdigest()) for i, f in iter_frames_at(self.video, reversed(want), gap)]
            self.assertEqual(a, b, f"gap {gap}")
        self.assertEqual([i for i, _ in iter_frames_at(self.video, [49, 70])], [49])     # past the end

    def test_motion_seek_gives_the_same_motion_values(self):
        out = []
        for seek in (False, True):
            cfg = Config.load(overrides={"shots": {"motion_seek": seek, "motion_pairs": 4, "motion_pair_s": 0.2}})
            _, ctx = make_ctx(self.root)
            out.append(ShotDetector(cfg, ctx).classify_all(self.video, self.shots, fps=25.0))
        self.assertEqual(out[0], out[1])
        self.assertEqual(set(out[0]), {"s01", "s02"})

    def test_reference_from_the_pose_pass_matches_the_reference_stage(self):
        pa, ca = self.pose()
        refs_a = pa.reference_frames(ca.run_id, self.shots, video=self.video)
        pb, cb = self.pose(batch=8, reference_in_pass=True)
        self.dets(pb, cb, with_reference=True)
        self.assertEqual(refs_a, pb.pass_refs)
        self.assertTrue(refs_a)
        self.assertEqual(ca.manifest["reference"], cb.manifest["reference"])
        for r in refs_a:
            self.assertEqual(json.loads(ca.paths.ref_json(r["shot_id"]).read_text()),
                             json.loads(cb.paths.ref_json(r["shot_id"]).read_text()))
            pa_png = cv2.imread(str(ca.paths.ref_png(r["shot_id"])))
            pb_png = cv2.imread(str(cb.paths.ref_png(r["shot_id"])))
            np.testing.assert_array_equal(pa_png, pb_png)

    def test_on_shot_done_is_called_per_shot_in_order(self):
        seen = []
        p, ctx = self.pose(reference_in_pass=True)
        p.estimate_body_shots(ctx.run_id, self.shots, video=self.video, with_reference=True,
                              on_shot_done=lambda s, path, ref: seen.append((s.shot_id, Path(path).is_file(),
                                                                             ref is not None)))
        self.assertEqual(seen, [("s01", True, True), ("s02", True, True)])

    def test_tracking_in_a_worker_process_gives_the_same_records(self):
        p, ctx = self.pose()
        dets = p.estimate_body_shots(ctx.run_id, self.shots, video=self.video)
        info = {"cfr": {"fps": 25.0}}
        ctx.manifest["ingest"] = info
        Tracker(p.cfg, ctx).track(ctx.run_id, self.shots[1], dets["s02"], None, fps=25.0, size=(320, 240),
                                  meta={"pose_model": "yolo26x-pose", "pose_variant": "x", "seed": 42})
        in_process = json.loads(json.dumps(ctx.manifest["tracking"]))
        assign_a = np.load(ctx.paths.assign("s02"))["assign"]
        job = {"run_id": ctx.run_id, "source": "public", "paths": ctx.paths, "cfg": p.cfg, "shot": self.shots[1],
               "det": dets["s02"], "ref": None, "fps": 25.0, "size": (320, 240),
               "meta": {"pose_model": "yolo26x-pose", "pose_variant": "x", "seed": 42}, "ingest": info}
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_track_worker_init) as ex:
            r = ex.submit(track_shot_job, job).result()
        self.assertEqual(no_seconds(r["tracking"]), no_seconds(in_process))
        np.testing.assert_array_equal(np.load(ctx.paths.assign("s02"))["assign"], assign_a)
        self.assertTrue(r["log"])

    def test_overlap_pose_needs_reference_in_pass(self):
        with self.assertRaises(AvError):
            Config.load(overrides={"track": {"overlap_pose": True}, "pose": {"reference_in_pass": False}})
        Config.load(overrides={"track": {"overlap_pose": True}, "pose": {"reference_in_pass": True}})


class ChunkedOverlayTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.cfg, self.ctx = make_ctx(self.root)
        self.video = ffmpeg_clip(self.ctx.paths.tmp_dir / "clip.mp4", seconds=6, rate=10, size="640x360")
        rid = self.ctx.run_id
        self.shots = [Shot(f"{rid}-s001", rid, 0, 25, 0.0, 2.5, "fixed"), Shot(f"{rid}-s002", rid, 25, 60, 2.5, 6.0,
                                                                                 "handheld")]
        rng = np.random.default_rng(3)
        T = 60
        wrists = np.stack([np.stack([320 + 80 * np.sin(np.arange(T) / 3), 250 + 40 * np.cos(np.arange(T) / 5)], 1),
                           np.stack([300 + 60 * np.cos(np.arange(T) / 4), 260 + rng.normal(0, 10, T)], 1)], 1)
        s = pose_series(T, angle=np.linspace(10, 70, T), wrists=wrists)
        s.frame_idx[:] = np.arange(T)
        from assemblyvision.analysis.motion import MotionAnalyzer
        sub = MotionAnalyzer(self.cfg).subject(s, self.shots[0])
        self.hud = overlay.overlay_series([sub], T + 1, 10.0, self.cfg, window_s=1.5)
        self.items = {}
        for sh in self.shots:            # two tracked people per frame
            n = sh.end_frame - sh.start_frame
            fi = np.repeat(np.arange(sh.start_frame, sh.end_frame, dtype=np.int32), 2)
            box = np.tile(np.array([[50, 40, 150, 330], [300, 30, 420, 340]], np.float32), (n, 1))
            kp = rng.uniform(40, 330, (2 * n, 17, 2)).astype(np.float32)
            det = {"det_version": np.int32(1), "frame_start": np.int32(sh.start_frame), "frame_count": np.int32(n),
                   "pose_variant": np.str_("x"), "frame_idx": fi, "bbox_xyxy": box,
                   "det_conf": np.full(2 * n, 0.8, np.float32), "body_xy": kp,
                   "body_conf": rng.uniform(0, 1, (2 * n, 17)).astype(np.float32)}
            np.savez(self.ctx.paths.det(sh.shot_id), **det)
            np.savez(self.ctx.paths.assign(sh.shot_id), assign=np.tile(np.array([1, 2], np.int64), n))
            self.items[sh.shot_id] = {"det": self.ctx.paths.det(sh.shot_id), "assign": self.ctx.paths.assign(sh.shot_id),
                                      "subject": 1}
        from assemblyvision.types import Step
        self.steps = [Step("Removal", 0.0, 3.0, "engine_a", 1.0), Step("Cleaning", 3.0, 6.0, "engine_a", 1.0)]

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, start, stop, out, hashes):
        return {"video": self.video, "shots": self.shots, "items": self.items, "fps": 10.0, "cfg": self.cfg,
                "steps": self.steps, "hud": self.hud, "start": start, "stop": stop, "out": out, "frame_hashes": hashes}

    def test_chunks_draw_exactly_the_frames_of_one_renderer(self):
        whole: list = []
        r = overlay.render_chunk(self.job(0, 60, self.ctx.paths.tmp_dir / "whole.mp4", whole))
        self.assertEqual((r["code"], r["frames"]), (0, 60))
        parts: list = []
        for a, b in overlay.chunk_bounds(0, 60, 3, [0, 25]):
            hashes: list = []
            overlay.render_chunk(self.job(a, b, self.ctx.paths.tmp_dir / f"p{a}.mp4", hashes))
            parts += hashes
        self.assertEqual(parts, whole)      # the hand reach map is replayed, so mid chunk frames match too

    def test_chunk_bounds_prefer_shot_starts(self):
        self.assertEqual(overlay.chunk_bounds(0, 600, 3, [0, 205, 410]), [(0, 205), (205, 410), (410, 600)])
        self.assertEqual(overlay.chunk_bounds(0, 600, 1, []), [(0, 600)])
        self.assertEqual(len(overlay.chunk_bounds(0, 40, 8, [])), 1)       # too short to split

    def test_parallel_overlay_file_has_every_frame(self):
        cfg = Config.load(overrides={"overlay": {"workers": 3, "encoder": "libx264"}})
        out = overlay.render_track_overlay(self.video, self.shots, self.items, self.ctx.paths.tracks_overlay, 10.0,
                                           cfg, self.ctx, steps=self.steps, hud=self.hud)
        cap = cv2.VideoCapture(str(out))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        self.assertEqual(n, 60)
        m = self.ctx.manifest["overlay"]
        self.assertEqual((m["frames"], m["workers"], m["encoder"]), (60, 2, "libx264"))   # >= 30 frames a chunk
        self.assertEqual(sorted(p.name for p in out.parent.iterdir()), ["tracks_overlay.mp4"])   # no parts left
        self.assertIn("concat", m["breakdown_s"])

    def test_encoder_command(self):
        c = overlay.encoder_cmd(640, 360, 10.0, {"crf": 20, "encoder": "h264_nvenc"}, Path("o.mp4"))
        self.assertEqual(c[c.index("-c:v") + 1], "h264_nvenc")
        c = overlay.encoder_cmd(640, 360, 10.0, {"crf": 20, "encoder": "libx264"}, Path("o.mp4"))
        self.assertEqual(c[c.index("-c:v") + 1], "libx264")


if __name__ == "__main__":
    unittest.main()


class ParallelShotsTest(unittest.TestCase):
    """shots.parallel: the pose pass starts before the shot table and gives the same files."""
    setUp, tearDown = SpeedOptionsTest.setUp, SpeedOptionsTest.tearDown
    pose, assertSameDets = SpeedOptionsTest.pose, SpeedOptionsTest.assertSameDets

    def later(self, after_calls: int):
        n = {"calls": 0}

        def shots_later(block: bool):
            n["calls"] += 1
            return self.shots if block or n["calls"] > after_calls else None
        return shots_later

    def run_pass(self, shots_later=None, **pose_cfg):
        p, ctx = self.pose(reference_in_pass=True, **pose_cfg)
        seen = []
        paths = p.estimate_body_shots(ctx.run_id, None if shots_later else self.shots, video=self.video,
                                      with_reference=True, shots_later=shots_later, n_total=50,
                                      on_shot_done=lambda s, path, ref: seen.append((s.shot_id, ref is not None)))
        dets = {sid: load_det(path) for sid, path in paths.items()}
        refs = {r["shot_id"]: (json.loads(ctx.paths.ref_json(r["shot_id"]).read_text()),
                               cv2.imread(str(ctx.paths.ref_png(r["shot_id"])))) for r in p.pass_refs}
        return dets, refs, seen, no_seconds(ctx.manifest["pose_body"]), ctx

    def test_same_files_whenever_the_shot_table_arrives(self):
        for batch in (1, 8):
            ref = self.run_pass(batch=batch)
            for after in (0, 5, 30, 10 ** 6):     # at once, inside shot 1, inside shot 2, only after the pass
                got = self.run_pass(self.later(after), batch=batch)
                self.assertSameDets(ref[0], got[0])
                self.assertEqual(set(ref[1]), set(got[1]))
                for sid in ref[1]:
                    self.assertEqual(ref[1][sid][0], got[1][sid][0])
                    np.testing.assert_array_equal(ref[1][sid][1], got[1][sid][1])
                self.assertEqual(ref[2], got[2])
                a, b = dict(ref[3]), dict(got[3])
                for m in (a, b):
                    m.pop("breakdown_s"), m.pop("pose_fps"), m.pop("infer_ms_mean")
                    m["per_shot"] = [{k: v for k, v in x.items() if k != "det_path"} for x in m["per_shot"]]
                self.assertEqual(a, b, f"batch {batch}, after {after}")

    def test_parallel_needs_reference_in_pass(self):
        with self.assertRaises(AvError):
            Config.load(overrides={"shots": {"parallel": True}, "pose": {"reference_in_pass": False},
                                   "track": {"overlap_pose": False}})
        Config.load(overrides={"shots": {"parallel": True}, "pose": {"reference_in_pass": True}})

    def test_thread_gives_the_same_shot_table(self):
        from assemblyvision.tools.run_pose import ParallelShots
        cfg, ctx = make_ctx(self.root)
        info = SimpleNamespace(fps=25.0, frame_count=50, width=320, height=240)
        direct = ShotDetector(cfg, ctx).detect(ctx.run_id, self.video, info)
        table = no_seconds(json.loads(json.dumps(ctx.manifest["shots"])))
        cfg2, ctx2 = make_ctx(self.root)
        ctx2.manifest.setdefault("stages", {})
        job = ParallelShots(cfg2, ctx2, None, ctx.run_id, self.video, info).start()
        got = job.result(True)
        self.assertEqual([(s.shot_id, s.start_frame, s.end_frame, s.shot_type) for s in got],
                         [(s.shot_id, s.start_frame, s.end_frame, s.shot_type) for s in direct])
        g = no_seconds(json.loads(json.dumps(ctx2.manifest["shots"])))
        for t in (table, g):
            t.pop("breakdown_s")
        self.assertEqual(g, table)
        st = ctx2.manifest["stages"]["shots"]
        self.assertEqual((st["status"], st["parallel_with"]), ("done", "pose_body"))
        self.assertIs(job.result(), got)
