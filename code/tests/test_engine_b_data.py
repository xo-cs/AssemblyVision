"""Engine B data (body frame features, YOLO like augmentation, cache build) and the domain gap metrics."""
import io
import json
import unittest
import zipfile
from pathlib import Path

import numpy as np

from assemblyvision.config import Config
from assemblyvision.datasets import domain_gap, engine_b, ha4m
from assemblyvision.datasets.remote import Pacer, open_remote_zip, read_member
from helpers import fake_root
from test_datasets import write_video
from test_fetch_dataset import Server, make_zip


def upright(T: int = 4, x0: float = 500.0, torso: float = 200.0) -> np.ndarray:
    """A standing person, mid hip at (x0, 800); every offset scales with the torso length."""
    k = torso / 200.0
    rel = np.zeros((17, 2))
    rel[:, 0] = (np.arange(17) - 8) * 3
    rel[:, 1] = -250
    rel[5], rel[6] = [60, -200], [-60, -200]          # shoulders
    rel[11], rel[12] = [40, 0], [-40, 0]              # hips
    rel[9] = [100, -100]                              # left wrist
    return np.repeat((np.array([x0, 800.0]) + k * rel)[None], T, axis=0)


class FeatureTest(unittest.TestCase):
    def test_body_frame_and_layout(self):
        xy = upright()
        f = engine_b.features(xy, np.full((4, 17), 0.9), image_h=1536.0)
        self.assertEqual(f.shape, (4, engine_b.N_FEATURES))
        self.assertEqual(engine_b.N_FEATURES, 55)
        body = f[:, :34].reshape(4, 17, 2)
        np.testing.assert_allclose(body[0, 5], [0.3, 1.0], atol=1e-6)         # +x = picture right when upright
        np.testing.assert_allclose(body[0, 11], [0.2, 0.0], atol=1e-6)        # hips on the x axis
        np.testing.assert_allclose(body[0, 9], [0.5, 0.5], atol=1e-6)
        self.assertTrue((f[:, 54] == 1).all())
        np.testing.assert_allclose(f[0, 51:53], [500 / 1536, 800 / 1536], atol=1e-6)
        self.assertAlmostEqual(float(f[0, 53]), 0.0, places=6)               # log(L / median L)

    def test_same_pose_anywhere_gives_the_same_body_features(self):
        a = engine_b.features(upright(x0=500, torso=200), np.full((4, 17), 0.9), 1536.0)
        b = engine_b.features(upright(x0=900, torso=100), np.full((4, 17), 0.9), 1536.0)
        np.testing.assert_allclose(a[:, :34], b[:, :34], atol=1e-6)

    def test_missing_torso_gives_an_invalid_frame(self):
        xy = upright()
        conf = np.full((4, 17), 0.9)
        conf[1, 11] = 0.0
        f = engine_b.features(xy, conf, 1536.0)
        self.assertEqual(f[:, 54].tolist(), [1, 0, 1, 1])
        self.assertTrue((f[1, :34] == 0).all())


class AugmentTest(unittest.TestCase):
    def test_conf_levels(self):
        a = engine_b.AugmentConfig()
        np.testing.assert_allclose(engine_b.yolo_like_conf(np.array([0.0, 0.333, 0.667]), a), [0.0, 0.4, 0.9])

    def test_jitter_dropout_and_noise_follow_the_settings(self):
        rng = np.random.default_rng(0)
        T = 4000
        xy = upright(T)
        kc = np.full((T, 17), 0.667)
        a = engine_b.AugmentConfig(jitter_torso=0.05, joint_dropout=0.1, frame_dropout=0.05, conf_noise=0.1)
        out, conf = engine_b.augment(xy, kc, a, rng)
        missing = np.isnan(out[..., 0])
        self.assertAlmostEqual(missing.all(axis=1).mean(), 0.05 + 0.95 * 0.1 ** 17, delta=0.01)
        self.assertAlmostEqual(missing[~missing.all(axis=1)].mean(), 0.1, delta=0.01)
        self.assertTrue((conf[missing] == 0).all())
        d = (out - xy)[~missing]
        sd = d.reshape(-1, 2)[:, 0].std()
        self.assertTrue(0.03 * 200 < sd < 0.07 * 200)                       # about 0.05 torso lengths of 200 px
        hip_sd = (out - xy)[:, 11, 0][~missing[:, 11]].std()
        nose_sd = (out - xy)[:, 0, 0][~missing[:, 0]].std()
        self.assertGreater(hip_sd, 3 * nose_sd)                               # COCO sigmas: hips 1.07, nose 0.26
        self.assertAlmostEqual(conf[~missing].mean(), 0.9, delta=0.02)
        self.assertTrue(((conf >= 0) & (conf <= 1)).all())

    def test_config_is_validated_and_recorded(self):
        cfg = Config.load(overrides={"engine_b": {"jitter_torso": 0.07}})
        self.assertEqual(engine_b.AugmentConfig.from_cfg(cfg).jitter_torso, 0.07)
        self.assertIn("jitter_torso = 0.07", cfg.to_toml())
        from assemblyvision.errors import AvError
        with self.assertRaises(AvError):
            Config.load(overrides={"engine_b": {"joint_dropout": 1.5}})


class BuildTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.repo = Path(self._tmp.name)
        self.root = self.repo / "data/public/datasets/ha4m"
        labels = [(f, 0 if f < 4 else 3, 0) for f in range(10)]
        write_video(self.root, "IDU001V001", labels, "000124702712")
        write_video(self.root, "IDU023V001", labels, "000099302712")

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_extends_and_skips(self):
        cfg = Config.load()
        loader = ha4m.HA4MLoader(cfg, repo_root=self.repo)
        logs = []
        idx = engine_b.build(loader, cfg, 1536.0, log=logs.append)
        self.assertEqual((idx["n_videos"], idx["n_frames"]), (2, 20))
        self.assertIn("2 videos built", logs[-1])
        engine_b.build(loader, cfg, 1536.0, log=logs.append)
        self.assertIn("0 videos built", logs[-1])
        write_video(self.root, "IDU002V001", [(f, 1, 0) for f in range(5)], "000124702712")
        (self.root / "labels" / "IDU003V001.csv").write_text("frame,action,repetition\n0,1,0\n")  # still fetching
        idx = engine_b.build(loader, cfg, 1536.0, log=logs.append)
        self.assertIn("1 videos built", logs[-1])
        self.assertEqual(sorted(idx["videos"]), ["IDU001V001", "IDU002V001", "IDU023V001"])
        cache = engine_b.cache_dir(self.root)
        clean = engine_b.load_video(cache, "IDU001V001")
        self.assertEqual(clean["features"].shape, (10, 55))
        self.assertEqual(int(clean["boundary"].sum()), 10)                    # change at 4, tol 6 frames
        aug = engine_b.load_video(cache, "IDU001V001", cfg, train=True, rng=np.random.default_rng(1))
        self.assertFalse(np.allclose(aug["features"], clean["features"]))
        self.assertEqual(json.loads((cache / "index.json").read_text())["settings"]["feature_version"], 1)


class DomainGapTest(unittest.TestCase):
    def frame(self, shift=0.0, found=True):
        k32 = np.zeros((32, 2))
        k32[:, 0] = np.linspace(400, 600, 32)
        k32[:, 1] = np.linspace(300, 900, 32)
        c32 = np.full(32, 0.667)
        kxy, kc = ha4m.kinect_to_coco(k32[None].astype(np.float32), c32[None].astype(np.float32))
        box = domain_gap.kinect_box(k32, c32)
        yb = np.array([box]) if found else np.array([[0, 0, 10, 10]])
        return {"k_xy": kxy[0], "k_conf": kc[0], "k_box": box, "y_boxes": yb,
                "y_xy": (kxy[0] + shift)[None], "y_conf": np.full((1, 17), 0.9)}

    def test_metrics(self):
        side = 600.0
        out = domain_gap.compare([self.frame(0.0), self.frame(0.04 * side / np.sqrt(2)), self.frame(found=False)])
        s = out["summary"]
        self.assertEqual((s["frames"], s["frames_with_kinect"], s["frames_found"]), (3, 3, 2))
        self.assertAlmostEqual(s["found_share"], 0.6667, places=3)
        self.assertEqual((s["pck_0.05"], s["pck_0.1"]), (1.0, 1.0))
        out = domain_gap.compare([self.frame(0.08 * side / np.sqrt(2))])
        self.assertEqual((out["summary"]["pck_0.05"], out["summary"]["pck_0.1"]), (0.0, 1.0))
        self.assertAlmostEqual(out["joints"][0]["mean_px"], 0.08 * side, delta=0.5)
        self.assertEqual(len(out["joints"]), 17)


class ReadMemberTest(unittest.TestCase):
    def test_one_request_per_member_and_crc(self):
        data = make_zip()
        srv = Server(data)
        zf, raw = open_remote_zip("http://x", opener=srv, buffer_size=1 << 10, pacer=Pacer(min_interval_s=0))
        info = next(i for i in zf.infolist() if i.filename.endswith("Labels.txt"))
        before = srv.requests
        self.assertEqual(read_member(raw, info), zipfile.ZipFile(io.BytesIO(data)).read(info.filename))
        self.assertEqual(srv.requests - before, 1)
        bad = bytearray(data)
        i = bad.index(b"0 0 0\n1 0 0")
        bad[i] = ord("9")
        zf2, raw2 = open_remote_zip("http://x", opener=Server(bytes(bad)), pacer=Pacer(min_interval_s=0))
        info2 = next(i for i in zf2.infolist() if i.filename.endswith("Labels.txt"))
        with self.assertRaises(zipfile.BadZipFile):
            read_member(raw2, info2)


if __name__ == "__main__":
    unittest.main()
