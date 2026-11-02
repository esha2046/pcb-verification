# Demo script & talking points (≈5 min)

## Live demo order
1. **Golden reference** - show the one reference board + component map (make_roi.py / config.json). "Only one reference needed."
2. **Good board** -> PASS 100 %. Open **Pipeline view**: ORB keypoint matches, RANSAC inliers, aligned image, difference map.
3. **Pull a part / shift one / drop a screw** (real board if possible) -> FAIL, overlay colours, per-component table with IoU / δ / SSIM.
4. **Video** upload (live removal) -> score-over-time chart drops when part is removed.
5. **evaluate.py results** + ablation table.
6. Swap reference/config for a second board (no retraining) -> "configurable".

## Results to quote (synthetic benchmark, 60 boards, random pose/light/noise; see out/evaluation_results.csv)
| | detector only | detector + verification |
|---|---|---|
| missing-part F1 | 0.95 | 0.95 |
| misplaced-part recall | 0.00 | 0.92 |
| extra-object recall | 0.00 | 1.00 |
| false-accept (defective passed) | 22.7 % | 0.0 % |
| false-reject (good failed) | 0.0 % | 0.0 % |
| board accuracy | 83.3 % | 100 % |
| latency | ~165-260 ms | ~185-280 ms (~3.5-5.5 FPS, single CPU core) |

Message: *detection alone confirms presence only; reference verification is what catches misplaced and extra parts* (= research gap #3 and #4).

## Be upfront (examiners respect this)
- Numbers above are on a **synthetic** board with simulated defects; real-board validation = Phase II / your own photos via evaluate on a folder.
- Default detector is template matching (works from 1 reference). YOLOv8 pipeline (dataset generator, training, ONNX benchmark, integration + tests) is implemented; trained weights are produced with `python train.py`.
- Limits: sensitive to large lighting/angle change; identical neighbouring parts; tiny SMD needs higher resolution / tiling.

## Mapping to your slides
Objectives 1-5 -> detect / completeness / spatial check / lightweight (CPU, open-source) / report.
Metrics slide -> evaluate.py.  Risks slide -> synthetic defect generation, augmentation, logging.  Edge claim -> benchmark_onnx.py.

## Real-photo results (public dataset photos; python fetch_real_data.py && python inspect_real.py)
One complete board = golden; 8 other real boards from 3 photos (different lighting/scale/white balance).
| board | verdict | note |
|---|---|---|
| mostly_ok_1, mostly_ok_2 | PASS 100 % | complete boards, same session as golden |
| missing_3 | FAIL 90 % | 2 flagged (C6/C7 area) - visually unpopulated pads |
| missing_2 | FAIL 65 % | bridge rectifier footprint empty -> flagged |
| missing_lots_1/2/3 | FAIL 20-60 % | IC1 / bridge / U$9 genuinely absent -> flagged |
| missing_1 | FAIL 40 % | appears populated; over-flagged (different white balance/scale) = false rejects |

Honest reading: reference verification works on real photos (good boards pass, absent ICs/bridges flagged), but small
parts (0402/0603 passives) are over-flagged when lighting/scale differ from the golden photo. That is the limitation of
template matching and exactly the motivation for the YOLOv8 backend (train on real labelled boards) in Phase II.
Source: github.com/aryan-programmer/pcb-fault-detection (test_images). Cite it; no ground-truth labels were provided,
so real-photo results are qualitative (visual inspection), not precision/recall.
