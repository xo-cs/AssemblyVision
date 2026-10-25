"""Track overlay header: drawn once per frame, at the bottom left (CPU)."""
import unittest
from pathlib import Path
from unittest import mock

import cv2

from assemblyvision.output import overlay
from assemblyvision.types import Shot, Step
from helpers import fake_root, ffmpeg_clip, make_ctx


class HeaderTest(unittest.TestCase):
    def test_header_once_bottom_left(self):
        tmp = fake_root()
        try:
            cfg, ctx = make_ctx(Path(tmp.name))
            video = ffmpeg_clip(ctx.paths.tmp_dir / "clip.mp4", seconds=1, rate=10, size="320x240")
            shots = [Shot(f"{ctx.run_id}-s001", ctx.run_id, 0, 10, 0.0, 1.0, "fixed")]
            steps = [Step("Removal", 0.0, 0.5, "engine_a"), Step("other", 0.5, 1.0, "engine_a")]
            calls = []
            real = cv2.putText

            def spy(img, text, org, *a, **k):
                calls.append((text, org, img.shape[0]))
                return real(img, text, org, *a, **k)
            with mock.patch.object(overlay.cv2, "putText", side_effect=spy):
                overlay.render_track_overlay(video, shots, {}, ctx.paths.tracks_overlay, 10.0, cfg, ctx, steps=steps)
            headers = [c for c in calls if c[0].startswith("shot ")]
            self.assertEqual(len(headers), 10)                    # once per frame, no second copy
            self.assertEqual(len(calls), 10)
            self.assertTrue(all(org[1] > h / 2 and org[0] < 40 for _, org, h in headers))
            self.assertIn("step: Removal", headers[0][0])
            self.assertIn("step: other", headers[-1][0])
            self.assertTrue(ctx.paths.tracks_overlay.is_file())
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
