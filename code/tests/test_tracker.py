"""Tracker on synthetic DET v1 files (CPU, no video needed)."""
import json
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.errors import AvError
from assemblyvision.kts import read_kts
from assemblyvision.perception.pose import save_det
from assemblyvision.perception.tracker import (SHORT, SUPPRESSED, Tracker, load_assign,
                                               suppress_duplicates)
from assemblyvision.types import Shot
from helpers import fake_root, make_ctx

FPS = 30.0
# a standing skeleton in box coordinates (0..1), COCO 17 order
_TEMPLATE = np.array([[.5, .08], [.45, .06], [.55, .06], [.4, .08], [.6, .08], [.3, .25], [.7, .25],
                      [.25, .42], [.75, .42], [.22, .58], [.78, .58], [.38, .58], [.62, .58],
                      [.38, .78], [.62, .78], [.38, .97], [.62, .97]], np.float32)


def person(x1, y1, w=60, h=160, conf=0.9):
    box = np.array([x1, y1, x1 + w, y1 + h], np.float32)
    kxy = _TEMPLATE * [w, h] + [x1, y1]
    return box, conf, kxy.astype(np.float32), np.full(17, 0.9, np.float32)


class DetBuilder:
    def __init__(self):
        self.rows = []

    def add(self, f, box, conf, kxy, kconf):
        self.rows.append((f, box, conf, kxy, kconf))

    def save(self, ctx, shot):
        r = self.rows
        rows = {"frame_idx": np.array([x[0] for x in r], np.int32),
                "bbox_xyxy": np.array([x[1] for x in r], np.float32).reshape(-1, 4),
                "det_conf": np.array([x[2] for x in r], np.float32),
                "body_xy": np.array([x[3] for x in r], np.float32).reshape(-1, 17, 2),
                "body_conf": np.array([x[4] for x in r], np.float32).reshape(-1, 17)}
        return save_det(ctx.paths.det(shot.shot_id), ctx, rows, shot.start_frame,
                        shot.end_frame - shot.start_frame, "l")


class TrackerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.cfg, self.ctx = make_ctx(Path(self._tmp.name))
        self.tr = Tracker(self.cfg, self.ctx)

    def tearDown(self):
        self._tmp.cleanup()

    def shot(self, n=1, start=0, frames=150):
        return Shot(f"{self.ctx.run_id}-s{n:03d}", self.ctx.run_id, start, start + frames,
                    start / FPS, (start + frames) / FPS, "fixed")

    def run_track(self, b, shot, **kw):
        det = b.save(self.ctx, shot)
        return self.tr.track(self.ctx.run_id, shot, det, fps=FPS, size=(640, 480), **kw)

    def test_two_walkers_keep_two_ids(self):   # TC-TRK-01
        s, b = self.shot(), DetBuilder()
        for f in range(150):
            b.add(f, *person(50 + 2 * f, 100))
            b.add(f, *person(50 + 2 * f, 300))
        recs = self.run_track(b, s)
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["frames_present"] >= 148 for r in recs))
        self.assertEqual(self.ctx.manifest["tracking"]["per_shot"][0]["short_tracks"], 0)

    def test_dropouts(self):   # TC-TRK-02
        s, b = self.shot(frames=240), DetBuilder()
        for f in range(240):
            if not (60 <= f < 75) and not (120 <= f < 180):   # 0.5 s and 2 s gaps
                b.add(f, *person(100 + f, 100))
        recs = self.run_track(b, s)
        self.assertEqual(len(recs), 2)   # same ID across 0.5 s, new ID after 2 s
        self.assertLessEqual(recs[0]["last_frame"], 120)
        self.assertGreaterEqual(recs[1]["first_frame"], 180)
        series, _ = read_kts(self.ctx.paths.kts(recs[0]["track_id"]))
        self.assertFalse(series.present[60:75].any())
        self.assertTrue(series.present[76:120].all())

    def test_long_gap_inside_track_notice(self):
        tr = Tracker(self.cfg, self.ctx)
        gaps = tr._long_gaps(np.array([1, 0, 0, 0, 0, 1, 0, 1], bool), buffer=3)
        self.assertEqual(gaps, [(1, 4)])

    def test_ids_restart_per_shot(self):   # TC-TRK-03
        b1, b2 = DetBuilder(), DetBuilder()
        s1, s2 = self.shot(1, 0, 90), self.shot(2, 90, 90)
        for f in range(90):
            b1.add(f, *person(100, 100))
            b2.add(90 + f, *person(100, 100))
        r1 = self.run_track(b1, s1)
        r2 = self.run_track(b2, s2)
        self.assertEqual(r1[0]["track_id"], f"{s1.shot_id}-t01")
        self.assertEqual(r2[0]["track_id"], f"{s2.shot_id}-t01")
        self.assertEqual(len(r1[0]["track_id"]), 33)

    def test_click_selects_and_outside_refused(self):   # TC-TRK-04
        s, b = self.shot(), DetBuilder()
        for f in range(150):
            b.add(f, *person(100, 100))
            b.add(f, *person(400, 100))
        ref = {"frame_idx": 10}
        recs = self.run_track(b, s, ref=ref, click_xy=(430, 200))
        subj = [r for r in recs if r["is_subject"]]
        self.assertEqual(len(subj), 1)
        self.assertEqual(subj[0]["subject_source"], "click")
        series, _ = read_kts(self.ctx.paths.kts(subj[0]["track_id"]))
        self.assertGreater(series.bbox_xyxy[10, 0], 300)
        with self.assertRaises(AvError) as cm:
            Tracker.pick_subject([series], ref, (5, 5), s.shot_id)
        self.assertEqual(cm.exception.code, "AV-TRK-003")

    def test_short_tracks_not_stored(self):   # TC-TRK-05
        s, b = self.shot(), DetBuilder()
        for f in range(150):
            b.add(f, *person(100, 100))
        for f in range(40, 55):   # 0.5 s visitor
            b.add(f, *person(400, 250))
        recs = self.run_track(b, s)
        self.assertEqual(len(recs), 1)
        stats = self.ctx.manifest["tracking"]["per_shot"][0]
        self.assertEqual((stats["tracks_created"], stats["short_tracks"]), (2, 1))
        assign = load_assign(self.ctx.paths.assign(s.shot_id))
        # ByteTrack confirms a track on its second frame, so the first row stays untracked
        self.assertEqual(int((assign == SHORT).sum()), 14)

    def test_arrays_cover_shot_without_interpolation(self):   # TC-TRK-06
        s, b = self.shot(start=30, frames=120), DetBuilder()
        for f in range(30, 150):
            if f % 10 != 0:
                b.add(f, *person(100, 100))
        recs = self.run_track(b, s)
        series, meta = read_kts(self.ctx.paths.kts(recs[0]["track_id"]))
        self.assertEqual(series.frame_idx.shape, (120,))
        self.assertEqual(int(series.frame_idx[0]), 30)
        missing = ~series.present
        self.assertTrue(np.isnan(series.body_xy[missing]).all())
        self.assertTrue(np.isnan(series.hand_xy).all())
        self.assertEqual((meta["kts_version"], meta["frame_count"], meta["tracker"]), (1, 120, "bytetrack"))

    def test_auto_subject_longest_track(self):
        s, b = self.shot(), DetBuilder()
        for f in range(150):
            b.add(f, *person(100, 100))
        for f in range(50):
            b.add(f, *person(400, 100))
        recs = self.run_track(b, s)
        subj = [r for r in recs if r["is_subject"]]
        self.assertEqual(len(subj), 1)
        self.assertEqual((subj[0]["frames_present"], subj[0]["subject_source"]), (150, "auto"))
        side = json.loads(self.ctx.paths.kts(subj[0]["track_id"]).with_suffix(".json").read_text())
        self.assertEqual((side["is_subject"], side["subject_source"]), (True, "auto"))
        self.assertEqual(self.ctx.manifest["tracking"]["per_shot"][0]["subject_source"], "auto")

    def test_no_auto_subject_below_coverage(self):
        s, b = self.shot(frames=300), DetBuilder()
        for f in range(60):   # 20 percent of the shot
            b.add(f, *person(100, 100))
        recs = self.run_track(b, s)
        self.assertFalse(any(r["is_subject"] for r in recs))
        self.assertIn("AV-POS-005", [n["code"] for n in self.ctx.manifest["notices"]])

    def test_partial_duplicate_suppressed_real_person_kept(self):
        tc = self.cfg["track"]
        full = person(100, 100)
        # upper body only: same joints, smaller box inside the full one
        part_box = np.array([100, 100, 160, 200], np.float32)
        part_kc = full[3].copy()
        part_kc[11:] = 0.05
        rows = [full, (part_box, 0.5, full[2].copy(), part_kc)]
        # a different worker standing behind: box inside, joints elsewhere
        behind = person(110, 110, w=40, h=110, conf=0.6)
        rows.append(behind)
        xyxy = np.stack([r[0] for r in rows])
        conf = np.array([r[1] for r in rows], np.float32)
        kxy = np.stack([r[2] for r in rows])
        kc = np.stack([r[3] for r in rows])
        dup = suppress_duplicates(xyxy, conf, kxy, kc, tc)
        self.assertEqual(dup.tolist(), [False, True, False])

    def test_duplicates_do_not_make_tracks(self):
        s, b = self.shot(), DetBuilder()
        for f in range(150):
            box, c, kxy, kc = person(100 + f, 100)
            b.add(f, box, c, kxy, kc)
            b.add(f, box + [0, 0, 0, -60], 0.4, kxy.copy(), kc.copy())   # partial copy
        recs = self.run_track(b, s)
        stats = self.ctx.manifest["tracking"]["per_shot"][0]
        self.assertEqual((len(recs), stats["duplicates_suppressed"]), (1, 150))
        assign = load_assign(self.ctx.paths.assign(s.shot_id))
        self.assertEqual(int((assign == SUPPRESSED).sum()), 150)

    def test_low_band_bridges_but_never_starts(self):
        s, b = self.shot(), DetBuilder()
        for f in range(150):
            b.add(f, *person(100 + f, 100, conf=0.9 if f < 60 or f >= 90 else 0.15))
        for f in range(150):
            b.add(f, *person(400, 300, conf=0.15))   # only weak: must not become a track
        recs = self.run_track(b, s)
        self.assertEqual(len(recs), 1)
        self.assertGreaterEqual(recs[0]["frames_present"], 148)


if __name__ == "__main__":
    unittest.main()
