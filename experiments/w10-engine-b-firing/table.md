# w10-engine-b-firing: Engine B firing rate on the HA4M test subjects (w10-engine-b-v0)

65 videos of the held out subjects (107.38 min, mean 99.1 s a video), 1167 true (action, repetition) changes = **10.87 per minute**. Probability pooled as Engine A pools it (engine_b.pool_s), divided by its maximum per video, peaks at 0.5, at least max(0.5 s, pool_s) apart. Model models/engine_b/w10-engine-b-v0.pt (hip calibration off).

| pool_s | peaks | peaks / min | F1@0.2 | F1@0.5 | F1@2 | P@0.2 | P@0.5 | P@2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1213 | 11.3 | 0.7471 | 0.8916 | 0.9479 | 0.7329 | 0.8747 | 0.9299 |
| 2 | 959 | 8.93 | 0.4704 | 0.6651 | 0.873 | 0.5214 | 0.7372 | 0.9677 |
| 5 | 414 | 3.86 | 0.067 | 0.1379 | 0.4731 | 0.128 | 0.2633 | 0.9034 |
| 10 | 229 | 2.13 | 0.0473 | 0.1203 | 0.2679 | 0.1441 | 0.3668 | 0.8166 |
| 30 | 95 | 0.88 | 0.0206 | 0.0396 | 0.1315 | 0.1368 | 0.2632 | 0.8737 |
| 60 | 74 | 0.69 | 0.0145 | 0.0306 | 0.1064 | 0.1216 | 0.2568 | 0.8919 |

Pool 0 is the raw model (11.3 peaks/min). A pooled window much longer than the video (HA4M videos are short) leaves about one peak per video.
