"""Pose labelling: frame sampler (public synthetic run), Label Studio exports, metrics
(pycocotools against our own OKS AP), the evaluation tool and the frame server."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from assemblyvision.datasets.mapping import COCO17
from assemblyvision.labelling import metrics, sampling, studio
from assemblyvision.labelling.server import serve
from assemblyvision.tools import eval_pose_labels, sample_frames
from helpers import fake_root
from labelling_helpers import frame_people, ls_box_task, make_run


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.repo = Path(self._tmp.name)
        self.run, self.manifest = make_run(self.repo)

    def tearDown(self):
        self._tmp.cleanup()

    def test_strata(self):
        th = sampling.Thresholds()
        table = {r["frame_idx"]: r for r in sampling.frame_table(self.manifest, self.run, th)}
        self.assertEqual(len(table), 100)
        self.assertEqual(table[10]["stratum"], "none")
        self.assertEqual(table[21]["stratum"], "occluded")
        self.assertEqual(table[31]["stratum"], "low_conf")
        self.assertEqual(table[41]["stratum"], "normal")
        self.assertEqual(table[60]["shot_type"], "handheld")

    def test_frames_command_is_stratified_seeded_and_stays_in_the_run(self):
        args = ["frames", "--run", str(self.run), "--n", "24", "--overlap", "4", "--min-gap-s", "0.1"]
        self.assertEqual(sample_frames.main(args, repo_root=self.repo), 0)
        out = self.run / "labelling"
        s = json.loads((out / "sample.json").read_text())
        self.assertEqual(s["frames"], 24)
        self.assertEqual(set(s["by_stratum"]), {"occluded", "low_conf", "none", "normal"})
        self.assertEqual(set(s["by_shot_type"]), {"fixed", "handheld"})
        self.assertEqual(s["overlap_frames"], 4)
        groups = [p["group"] for p in s["picks"]]
        self.assertEqual(groups.count("AB"), 4)
        self.assertLessEqual(abs(groups.count("A") - groups.count("B")), 1)
        self.assertEqual(len(list((out / "frames").glob("*.png"))), 24)
        tasks = json.loads((out / "tasks_boxes.json").read_text())
        self.assertTrue(tasks[0]["data"]["image"].startswith("http://127.0.0.1:8081/f"))
        self.assertTrue((out / "boxes.xml").is_file() and (out / "keypoints.xml").is_file())
        first = [p["frame_idx"] for p in s["picks"]]
        self.assertEqual(sample_frames.main(args, repo_root=self.repo), 2)          # no silent replace
        (out / "frames" / "f9999999.png").write_bytes(b"old")
        self.assertEqual(sample_frames.main(args + ["--replace"], repo_root=self.repo), 0)
        self.assertFalse((out / "frames" / "f9999999.png").exists())
        self.assertEqual(len(list((out / "frames").glob("*.png"))), 24)
        self.assertEqual([p["frame_idx"] for p in json.loads((out / "sample.json").read_text())["picks"]], first)

    def test_outside_a_run_folder_is_refused(self):
        other = self.repo / "data/public/elsewhere"
        other.mkdir(parents=True)
        (other / "manifest.json").write_text("{}")
        self.assertEqual(sample_frames.main(["frames", "--run", str(other)], repo_root=self.repo), 2)

    def test_crops_and_export_area(self):
        sample_frames.main(["frames", "--run", str(self.run), "--n", "12", "--overlap", "2", "--min-gap-s", "0.1"],
                           repo_root=self.repo)
        picks = json.loads((self.run / "labelling/sample.json").read_text())["picks"]
        f = next(p["frame_idx"] for p in picks if p["stratum"] == "occluded")
        export = [ls_box_task(f, [[50, 20, 110, 200], [70, 20, 130, 200]], "asel@x")]
        path = self.run / "labelling/export_boxes.json"
        path.write_text(json.dumps(export))
        self.assertEqual(sample_frames.main(["crops", "--run", str(self.run), "--boxes", str(path)],
                                            repo_root=self.repo), 0)
        tasks = json.loads((self.run / "labelling/tasks_keypoints.json").read_text())
        self.assertEqual(len(tasks), 2)
        self.assertTrue((self.run / "labelling/frames/crops" / Path(tasks[0]["data"]["image"]).name).is_file())
        wrong = self.repo / "data/qisens/export_boxes.json"                    # other data area
        wrong.write_text(json.dumps(export))
        self.assertEqual(sample_frames.main(["crops", "--run", str(self.run), "--boxes", str(wrong)],
                                            repo_root=self.repo), 2)

    def test_crop_window(self):
        x0, y0, x1, y1, s = sampling.crop_window([100, 50, 140, 210], 320, 240)
        self.assertTrue(0 <= x0 < 100 and 140 < x1 <= 320 and y0 == 0 and y1 == 240)
        self.assertAlmostEqual(s, 640 / max(x1 - x0, y1 - y0))


def person_gt(seed: int, x: float = 40.0) -> dict:
    rng = np.random.default_rng(seed)
    xy = np.stack([rng.uniform(x + 5, x + 55, 17), rng.uniform(25, 195, 17)], axis=1)
    vis = [2] * 15 + [1, 0]
    return {"box": [x, 20.0, x + 60, 200.0], "xy": xy.tolist(), "vis": vis, "labeller": "asel"}


def frames_fixture(shift: float = 0.0, drop: bool = False, extra: bool = False, seeds=range(10)) -> dict:
    frames = {}
    for i in seeds:
        g = person_gt(i)
        p = {"box": g["box"], "score": 0.9 - 0.01 * i, "xy": (np.asarray(g["xy"]) + shift).tolist()}
        preds = [] if drop and i % 2 else [p]
        if extra:
            preds.append({"box": [250, 20, 300, 200], "score": 0.5, "xy": [[270, 100]] * 17})
        frames[f"f{i}"] = {"gt": [g], "pred": preds}
    return frames


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        f = frames_fixture()
        own, coco = metrics.oks_ap_own(f), metrics.oks_ap_coco(f)
        self.assertEqual((own["AP"], coco["AP"]), (1.0, 1.0))
        self.assertEqual(metrics.pck(f)["mean"], 1.0)
        d = metrics.detection_pr(f, 0.25)
        self.assertEqual((d["precision"], d["recall"]), (1.0, 1.0))

    def test_own_ap_matches_pycocotools(self):
        rng = np.random.default_rng(3)
        for case in range(6):
            f = frames_fixture(shift=float(rng.uniform(0, 12)), drop=case % 2 == 1, extra=case >= 3)
            for k, fr in enumerate(f.values()):                    # varied errors per person
                for p in fr["pred"]:
                    p["xy"] = (np.asarray(p["xy"]) + rng.normal(0, 2 + k, (17, 2))).tolist()
            own, coco = metrics.oks_ap_own(f), metrics.oks_ap_coco(f)
            for key in ("AP", "AP50", "AP75"):
                self.assertAlmostEqual(own[key], coco[key], delta=1e-3, msg=f"case {case} {key}: {own} {coco}")

    def test_errors_lower_the_scores(self):
        shifted = frames_fixture(shift=15.0)                                # 21 px > 0.1 x 180 px
        self.assertLess(metrics.pck(shifted)["mean"], 1.0)
        self.assertLess(metrics.oks_ap_coco(shifted)["AP"], 1.0)
        d = metrics.detection_pr(frames_fixture(drop=True), 0.25)
        self.assertEqual(d["recall"], 0.5)
        d = metrics.detection_pr(frames_fixture(extra=True), 0.25)
        self.assertEqual((d["precision"], d["recall"]), (0.5, 1.0))
        d = metrics.detection_pr(frames_fixture(extra=True), 0.6)          # the extra one is below 0.6
        self.assertEqual(d["precision"], 1.0)

    def test_unlabelled_person_is_ignored_in_ap(self):
        f = frames_fixture()
        f["f0"]["gt"].append({"box": [250, 20, 300, 200], "xy": [[0, 0]] * 17, "vis": [0] * 17, "labeller": None})
        f["f0"]["pred"].append({"box": [250, 20, 300, 200], "score": 0.95, "xy": [[270, 100]] * 17})
        own, coco = metrics.oks_ap_own(f), metrics.oks_ap_coco(f)
        self.assertEqual((own["AP"], coco["AP"]), (1.0, 1.0))

    def test_agreement(self):
        a = person_gt(1)
        b = {**a, "xy": (np.asarray(a["xy"]) + 1.0).tolist()}
        ag = metrics.agreement([(a, a), (a, b)])
        self.assertEqual(ag["people"], 2)
        self.assertGreater(ag["mean_oks"], 0.9)
        self.assertEqual(ag["visibility_agreement"], 1.0)
        bx = metrics.box_agreement([([[0, 0, 10, 10]], [[1, 0, 11, 10]])])
        self.assertEqual(bx["matched"], 1)


class StudioAndEvalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.repo = Path(self._tmp.name)
        self.run, self.manifest = make_run(self.repo)

    def tearDown(self):
        self._tmp.cleanup()

    def kp_task(self, frame_idx: int, box_id: int, box, xy, labeller: str, x0=0, y0=0, s=1.0, occluded=()):
        res = []
        for k, (x, y) in enumerate(xy):
            name = COCO17[k] + (studio.OCCLUDED if k in occluded else "")
            res.append({"type": "keypointlabels", "original_width": 640, "original_height": 640,
                        "value": {"x": (x - x0) * s / 640 * 100, "y": (y - y0) * s / 640 * 100, "width": 0.5,
                                  "keypointlabels": [name]}})
        return {"data": {"frame_name": f"f{frame_idx:07d}.png", "frame_idx": frame_idx, "box_id": box_id, "box": box,
                         "crop_x0": x0, "crop_y0": y0, "crop_scale": s, "overlap": True},
                "annotations": [{"completed_by": {"id": 2, "email": labeller}, "result": res}]}

    def test_exports_to_frame_pixels(self):
        box = [40, 20, 100, 200]
        xy = [[50 + k, 30 + 5 * k] for k in range(17)]
        recs = studio.parse_keypoints([self.kp_task(11, 0, box, xy, "kazi@x", x0=10, y0=5, s=2.0, occluded=(3,))])
        np.testing.assert_allclose(recs[0]["xy"], xy, atol=1e-6)
        self.assertEqual(recs[0]["vis"][3], 1)
        self.assertEqual(recs[0]["vis"][0], 2)
        self.assertEqual(recs[0]["labeller"], "kazi@x")
        boxes = studio.parse_boxes([ls_box_task(11, [box], "asel@x")])
        np.testing.assert_allclose(boxes[0]["boxes"][0], box, atol=1e-6)
        t = ls_box_task(11, [box], "x")
        t["annotations"][0]["completed_by"] = 3                                  # Label Studio 1.23 export
        self.assertEqual(studio.parse_boxes([t])[0]["labeller"], "3")
        self.assertEqual(studio.parse_boxes([t], {"3": "kazi"})[0]["labeller"], "kazi")

    def test_skipped_annotation_is_left_out(self):
        t = ls_box_task(11, [[1, 1, 5, 5]], "a")
        t["annotations"][0]["result"].append({"type": "choices", "value": {"choices": ["frame unusable"]}})
        self.assertEqual(studio.parse_boxes([t]), [])

    def test_eval_tool_on_the_run_detections(self):
        """Labels equal to the DET rows of frames 11 and 41: perfect scores; the overlap
        crop labelled by a second person gives the agreement."""
        lab = self.run / "labelling"
        lab.mkdir()
        boxes, kps = [], []
        for f in (11, 41):
            box, conf, kp, kc = frame_people(f)[0]
            boxes.append(ls_box_task(f, [box.tolist()], "asel@x"))
            kps.append(self.kp_task(f, 0, box.tolist(), kp.tolist(), "asel@x"))
        kps.append(self.kp_task(11, 0, frame_people(11)[0][0].tolist(), (frame_people(11)[0][2] + 2).tolist(), "kazi@x"))
        (lab / "boxes.json").write_text(json.dumps(boxes))
        (lab / "kp.json").write_text(json.dumps(kps))
        rc = eval_pose_labels.main(["--run", str(self.run), "--boxes", str(lab / "boxes.json"),
                                    "--keypoints", str(lab / "kp.json")], repo_root=self.repo)
        self.assertEqual(rc, 0)
        res = json.loads((lab / "eval_pose.json").read_text())
        self.assertEqual((res["detection"]["precision"], res["detection"]["recall"]), (1.0, 1.0))
        self.assertEqual(res["oks_ap"]["pycocotools"]["AP"], 1.0)
        self.assertTrue(res["oks_ap"]["same"])
        self.assertEqual(res["pck"]["mean"], 1.0)
        self.assertEqual(res["agreement"]["keypoints"]["people"], 1)
        self.assertEqual(res["labellers"], ["asel@x"])
        self.assertIn("PCK@0.1", (lab / "eval_pose.md").read_text())
        wrong = self.repo / "data/qisens/kp.json"
        wrong.write_text("[]")
        self.assertEqual(eval_pose_labels.main(["--run", str(self.run), "--boxes", str(lab / "boxes.json"),
                                                "--keypoints", str(wrong)], repo_root=self.repo), 2)


class FrameServerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.dir = Path(self._tmp.name) / "frames"
        (self.dir / "crops").mkdir(parents=True)
        (self.dir / "f0000001.png").write_bytes(b"\x89PNG fake")
        (self.dir / "crops" / "f0000001_b00.png").write_bytes(b"\x89PNG crop")
        (self.dir / "notes.txt").write_text("secret")
        (Path(self._tmp.name) / "outside.png").write_bytes(b"\x89PNG out")
        self.srv, self.counts = serve(self.dir, port=0)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self._tmp.cleanup()

    def get(self, path: str):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5)
            return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), b""

    def test_png_with_no_store_headers(self):
        for path, body in (("/f0000001.png", b"\x89PNG fake"), ("/crops/f0000001_b00.png", b"\x89PNG crop")):
            status, headers, data = self.get(path)
            self.assertEqual((status, data), (200, body))
            self.assertIn("no-store", headers["Cache-Control"])
            self.assertEqual(headers["Pragma"], "no-cache")
            self.assertEqual(headers["Content-Disposition"], "inline")
            self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:8080")

    def test_everything_else_is_refused(self):
        for path in ("/", "/crops/", "/notes.txt", "/../outside.png", "/%2e%2e/outside.png", "/crops/../../outside.png"):
            status, headers, _ = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertIn("no-store", headers["Cache-Control"])
        self.assertEqual(self.counts["served"], 0)

    def test_only_loopback(self):
        with self.assertRaises(ValueError):
            serve(self.dir, host="0.0.0.0", port=0)


if __name__ == "__main__":
    unittest.main()
