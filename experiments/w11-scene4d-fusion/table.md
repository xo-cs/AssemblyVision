# w11-scene4d-fusion

Fusion of Pi3 improved and VGGT-1B (rule w11-scene4d-fusion, written before the runs): public process_480p30, fixed s021_p1 and handheld s024_p1 at 6 fps, chunks of 80, RTX A4000, one session, one measure script. Coverage and holes are the mean of both clips; the time is the larger clip, without writing the raw test points.

| candidate | licence | fixed camera rotation deg | background spread | cloud noise | coverage (higher is better) | holes behind the person | s per video min | peak MB | ranks | sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pi3 improved | BSD-2-Clause | 1.13 | 0.0308 | 0.00055 | 0.5497 | 0.4225 | 321.7 | 10068 | 2 + 2 + 2 + 2 + 3 | 11 |
| VGGT-1B | CC BY-NC 4.0 | 0.93 | 0.0272 | 0.00059 | 0.6544 | 0.1838 | 387.3 | 13956 | 1 + 1 + 3 + 1 + 1 | 7 |
| Fusion | CC BY-NC 4.0 | 1.13 | 0.0399 | 0.00045 | 0.4214 | 0.3332 | 419.2 | 13956 | 2 + 3 + 1 + 3 + 2 | 11 |

**Result: the fusion does not win** (sum 11, tied with Pi3 improved; VGGT-1B 7). It has the lowest cloud noise (0.00045) but the lowest coverage and a higher background spread than either model, because its strict filter for voxels seen by one model removes much of the surface. In the screenshots its handheld cloud has ragged edges. By the rule Pi3 improved stays as scene4d.model.

VGGT-1B has the lowest sum here (better rotation, spread, coverage and holes; worse noise). The rule of this experiment only decides on the fusion, so this is not a switch. The coverage and holes measures have no occlusion test and depend on a focal length estimated from each model's points, and the handheld screenshot of Pi3 looks more complete than its coverage (0.645 against 0.839) suggests, so these two numbers need care before they are used to choose VGGT.

Screenshots: logs/ui_shots/scene4d_fusion_{fixed,handheld}.png (Pi3 improved, VGGT-1B, fusion).
