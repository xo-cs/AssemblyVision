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

**After, once the prewarm has finished (two more measurements, 90 s after the start):** first page 0.74 and 0.71 s;
interactive page 2.74 and 1.17 s; switch to Results 0.29 and 0.17 s; open a run 1.03 and 1.20 s, again 1.24 and 1.25 s;
Movement tabs 3D 0.30 and 0.32 s, 4D 0.36 and 0.41 s, 2D 0.17 s; video start 0.05 s. So the 2.26 s first page above was
the prewarm thread still running. The interactive page varies (1.2 to 2.7 s) because each measurement starts with an
empty browser cache and loads about 160 script files. Targets met: every click within 1 s, a run opens within 2 s.

## Round 2 (2026-11-27), after Khurshid's flagship measurement

Flagship QiSens run, measured by Khurshid with scripts/ui_timing.py: first page 0.68 s, interactive 1.10 s, Results 0.19 s,
**open the run 10.90 s the first time** and 1.27 s the second, tabs 3D 0.24 s, 4D 0.34 s, 2D 0.12 s, video 0.05 s.

Likely cause of the slow first open: the per run cache keyed on every file of the run, so a new objects.npz (from
detect_objects after the app start) threw away the warm Results view, and the prewarm only ran at the start. Fixes:
each view is keyed only on the files it reads; a keep warm loop refills the listed runs every minute (and logs a run
that takes over 1 s or fails); the floor maps load only when their section is opened; the 3D and 4D player files are
served as compact copies (4D positions as int16 with an offset and a scale, largest error 0.02 mm on this run; NaN kept;
gzip). The event stream (text/event-stream) cannot be gzipped by Starlette, so it was made smaller instead.

Public run after these fixes: open a run 1.06 s, again 1.21 s; tabs 0.16 to 0.28 s; Gradio events 0.82 MB (was 1.28);
3D and 4D data 3.92 MB (was 7.36); the whole run's 4D files 120.8 MB to 62.4 MB, 3D files 3.5 MB to 0.67 MB. No 3D or 4D
file is loaded before its tab is opened (0 requests). The flagship re-measurement is Khurshid's.
