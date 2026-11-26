# w12-train-domain

Freeze: validation-freeze-1 (866fba2), ran on cf152c7, method unchanged: True. B zero shot, untuned.

**Clip classification:** accuracy 92.86 % over 14 labelled segments.

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| Track and Ballast Maintenance | 6 | 0.86 | 1.00 | 0.92 |
| Equipment and Machinery Coordination | 2 | 1.00 | 1.00 | 1.00 |
| Washing and Cleaning | 2 | 1.00 | 1.00 | 1.00 |
| Rolling Stock and Coupler Work | 1 | 1.00 | 1.00 | 1.00 |
| Metal Fabrication and Grinding | 1 | 1.00 | 1.00 | 1.00 |
| Locomotive Firing and Fueling | 2 | 1.00 | 0.50 | 0.67 |

Confusion matrix (rows true, columns predicted):

| | Track and Ballast Maintenance | Equipment and Machinery Coordination | Washing and Cleaning | Rolling Stock and Coupler Work | Metal Fabrication and Grinding | Locomotive Firing and Fueling |
| --- | --- | --- | --- | --- | --- | --- |
| Track and Ballast Maintenance | 6 | 0 | 0 | 0 | 0 | 0 |
| Equipment and Machinery Coordination | 0 | 2 | 0 | 0 | 0 | 0 |
| Washing and Cleaning | 0 | 0 | 2 | 0 | 0 | 0 |
| Rolling Stock and Coupler Work | 0 | 0 | 0 | 1 | 0 | 0 |
| Metal Fabrication and Grinding | 0 | 0 | 0 | 0 | 1 | 0 |
| Locomotive Firing and Fueling | 1 | 0 | 0 | 0 | 0 | 1 |

**Joined video (seed 20261121):** frame accuracy 89.69, edit 90.91, F1@10 95.24, F1@25 85.71, F1@50 85.71, boundary MAE 2.43 s.

## Reading the result (written after the run; the numbers above are as the tool wrote them)

- **Clip test: 13 of 14 clips correct (92.86 %).** Chance is 16.7 % (1 of 6 classes); always naming the most frequent
  class gives 42.9 %. The one error: 17352127 (Locomotive Firing and Fueling, 7.9 s, the shortest kind of clip) was named
  Track and Ballast Maintenance, with the smallest lead of any clip but one (1.03).
- **Joined video: frame accuracy 89.69, edit 90.91, F1@10/25/50 95.24 / 85.71 / 85.71, boundary MAE 2.43 s**, with no
  order and no number of segments given. The decoder found 10 of the 11 true segments. It missed the same short clip
  (192 to 200 s, predicted as Track). For 19497506 (Equipment and Machinery Coordination, 53 to 75 s) it gave only 5 s to
  Equipment and the rest to Track; that clip also had the smallest lead in the clip test (0.90), so it was correct there
  only by a small margin.
- **Limits:** 14 clips, so one clip is about 7 points. One annotator (Khurshid), no agreement figure. The class table was
  made by a model that watched these clips, so the classes fit this set. Track and Ballast Maintenance is 6 of the 14
  clips, and both errors went to it.
- **Run history:** one pass. It was stopped after 3 clips for the mentor meeting (2026-11-26) and continued from clip 4
  with the same command; no clip was run twice and nothing was tuned.
