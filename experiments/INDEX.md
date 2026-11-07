# Experiments

One line per experiment (public data only; QiSens rows only as approved summaries).
Regenerate any table with `python -m assemblyvision.tools.exp table <id> ...`.

| id | date | title | per video: full run median s, pose fps, mean GPU util %, peak GPU MB | verdict |
| --- | --- | --- | --- | --- |
| [w10-b16](w10-b16/table.md) | 2026-11-06 | pose batch 16, FP32 | process_480p30.mp4: 175.8 s, 108.2 fps, 44.4 %, 2717 MB<br>12981262_1920_1080_25fps.mp4: 62.5 s, 84.3 fps, 21.6 %, 2717 MB | different |
| [w10-b32](w10-b32/table.md) | 2026-11-06 | pose batch 32, FP32 | process_480p30.mp4: 177.0 s, 106.7 fps, 43.8 %, 4441 MB<br>12981262_1920_1080_25fps.mp4: 63.0 s, 84.7 fps, 22.9 %, 4441 MB | different |
| [w10-b8](w10-b8/table.md) | 2026-11-06 | pose batch 8, FP32 | process_480p30.mp4: 176.7 s, 107.2 fps, 44.3 %, 1799 MB<br>12981262_1920_1080_25fps.mp4: 63.8 s, 85.2 fps, 21.4 %, 1799 MB | different |
| [w10-b8-fp16](w10-b8-fp16/table.md) | 2026-11-06 | pose batch 8, FP16 | process_480p30.mp4: 156.4 s, 142.6 fps, 32.1 %, 1455 MB<br>12981262_1920_1080_25fps.mp4: 63.5 s, 94.0 fps, 13.2 %, 1455 MB | different |
| [w10-b8-notf32](w10-b8-notf32/table.md) | 2026-11-07 | pose batch 8, strict FP32 (TF32 off) | process_480p30.mp4: 284.2 s, 50.0 fps, 66.0 %, 6685 MB<br>12981262_1920_1080_25fps.mp4: 81.8 s, 45.2 fps, 39.7 %, 6685 MB | different |
| [w10-best](w10-best/table.md) | 2026-11-06 | equivalent winners combined (batch 1, FP32) | process_480p30.mp4: 192.7 s, 66.4 fps, 46.6 %, 1881 MB<br>12981262_1920_1080_25fps.mp4: 63.1 s, 60.3 fps, 27.0 %, 2020 MB | equivalent |
| [w10-best-b8](w10-best-b8/table.md) | 2026-11-07 | proposal (speed_proposed.toml) at pose batch 8, FP32 | process_480p30.mp4: 134.7 s, 108.4 fps, 54.9 %, 2793 MB<br>12981262_1920_1080_25fps.mp4: 50.2 s, 94.3 fps, 28.8 %, 2932 MB | different |
| [w10-nvdec](w10-nvdec/table.md) | 2026-11-06 | ffmpeg NVDEC reader (batch 1) | process_480p30.mp4: 230.0 s, 67.6 fps, 42.1 %, 1214 MB<br>12981262_1920_1080_25fps.mp4: 72.4 s, 58.2 fps, 23.1 %, 1305 MB | different |
| [w10-ovl-nvenc](w10-ovl-nvenc/table.md) | 2026-11-06 | overlay encoder h264_nvenc | process_480p30.mp4: 223.0 s, 66.8 fps, 42.7 %, 1206 MB<br>12981262_1920_1080_25fps.mp4: 64.2 s, 62.1 fps, 26.5 %, 1241 MB | equivalent |
| [w10-ovl-off](w10-ovl-off/table.md) | 2026-11-06 | overlay off | process_480p30.mp4: 186.0 s, 68.1 fps, 50.9 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 46.2 s, 61.2 fps, 36.5 %, 981 MB | equivalent |
| [w10-ovl-par4](w10-ovl-par4/table.md) | 2026-11-06 | overlay in 4 parallel chunks (libx264) | process_480p30.mp4: 218.7 s, 67.7 fps, 43.2 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 70.0 s, 60.4 fps, 23.1 %, 981 MB | equivalent |
| [w10-prefetch](w10-prefetch/table.md) | 2026-11-06 | threaded prefetch reader (batch 1) | process_480p30.mp4: 227.9 s, 68.8 fps, 41.5 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 70.5 s, 62.2 fps, 23.2 %, 981 MB | equivalent |
| [w10-qisens](w10-qisens/table.md) | 2026-11-07 | QiSens video (9.9 min), 3 repeats each: reference vs speed_proposed.toml (approved summary) | w10-qisens-ref: 392.2 s, 67.4 fps, 41.1 %, 981 MB<br>w10-qisens-best: 340.7 s, 68.1 fps, 48.0 %, 1881 MB | equivalent |
| [w10-ref](w10-ref/table.md) | 2026-11-06 | reference: YOLO26 x, batch 1, FP32, current defaults | process_480p30.mp4: 235.7 s, 65.3 fps, 39.9 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 71.7 s, 60.3 fps, 22.6 %, 981 MB | reference |
| [w10-ref-notf32](w10-ref-notf32/table.md) | 2026-11-07 | reference in strict FP32 (TF32 off), batch 1 | process_480p30.mp4: 276.9 s, 51.9 fps, 62.6 %, 913 MB<br>12981262_1920_1080_25fps.mp4: 78.5 s, 49.5 fps, 37.1 %, 913 MB | different |
| [w10-refpass](w10-refpass/table.md) | 2026-11-06 | reference frames from the pose pass | process_480p30.mp4: 217.9 s, 66.3 fps, 40.6 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 71.7 s, 58.3 fps, 23.1 %, 981 MB | equivalent |
| [w10-shots-ds](w10-shots-ds/table.md) | 2026-11-07 | cut detection downscale 8 (automatic: 3 at 480p, 7 at 1080p) | process_480p30.mp4: 233.3 s, 66.2 fps, 40.3 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 71.0 s, 60.5 fps, 22.8 %, 981 MB | equivalent |
| [w10-shots-seek](w10-shots-seek/table.md) | 2026-11-06 | shots motion pairs read by seeking | process_480p30.mp4: 234.4 s, 65.7 fps, 40.6 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 75.2 s, 59.5 fps, 21.8 %, 981 MB | equivalent |
| [w10-trackoverlap](w10-trackoverlap/table.md) | 2026-11-06 | tracking in a worker during pose (with reference from the pass) | process_480p30.mp4: 209.8 s, 66.3 fps, 42.2 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 72.8 s, 59.4 fps, 22.6 %, 981 MB | equivalent |
