# w12-ui-speed

Web app speed in headless Chrome through the real app code (test copy, public run run_20261025_063706_7fc4, 22 shot parts with 3D and 4D), on the server itself (no network delay), scripts/ui_timing.py. Before: commit 0be603e. After: per run caches, prewarm of the listed runs, gzip of the page, scripts and JSON (never video, files or the event stream), inline images as lossless WebP. Both measured just after an app start (cold).

| measure | before s | after s |
| --- | --- | --- |
| first page | 1.22 | 2.26 |
| interactive page | 2.45 | 2.36 |
| switch to Results | 0.43 | 0.22 |
| open a run (page load to filled Results) | 9.61 | 1.14 |
| open the run again | - | 1.26 |
| Movement tab 3D | 0.59 | 0.30 |
| Movement tab 4D | 0.32 | 0.33 |
| Movement tab 2D | 0.26 | 0.18 |
| overlay video starts | 0.06 | 0.05 |

| payload | before MB | after MB |
| --- | --- | --- |
| 3D/4D data | 7.36 | 7.36 |
| Gradio events and data | 2.16 | 1.28 |
| file | 0.18 | 0.18 |
| other | 0.00 | 0.03 |
| page | 0.57 | 0.09 |
| scripts and styles | 0.53 | 0.54 |
| video | 9.70 | 8.65 |

Before, 'open the run again' was not measured. The video already used range requests (HTTP 206, accept-ranges bytes) and starts in 0.05 s. Payload sizes depend on what the script clicks, so the video and 4D rows show what was loaded in this session, not a whole file. The 4D tab loads one shot part at a time (7.1 MB here); on a 100 Mbit/s network that adds about 0.6 s. The flagship QiSens run is measured by Khurshid with the same script.
