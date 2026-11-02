# Vision-Based Automated Assembly Verification & Missing-Component Detection (PCB)
RVCE · CCVR · Summer Internship Review-1 -> working prototype

Reference-guided, explainable inspection from **one golden board**; YOLOv8 backend for the trained model.

## Pipeline (matches slide 13)
1. Pre-process (resize + Gaussian denoise)  ->  2. **ORB + RANSAC homography** alignment
3. Detection: `TemplateDetector` (1 reference, no training) **or** `YoloDetector` (fine-tuned YOLOv8n)
4. **IoU + normalised centroid offset** (+ SSIM) vs reference ROIs  ->  correct / misplaced / missing / extra
5. **Assembly Completeness Score** C = correct / expected x 100 %  (PASS iff C = 100 % and no extras)
6. Streamlit dashboard (image / webcam / video), annotated overlay, per-component table, pipeline view, CSV/JSON logs

## Quick start
    pip install -r requirements.txt
    python make_demo.py                       # synthetic golden board + test boards + videos
    streamlit run app.py                      # dashboard
    python inspect_cli.py data/test_mixed.png
    python inspect_cli.py data/video_live_removal.mp4

## Evaluation / ablation (slide 16 validation plan)
    python evaluate.py --n 60                 # -> out/evaluation_results.csv
    python evaluate.py --n 60 --weights weights_best.pt     # same with trained YOLO
Reports per-defect precision/recall/F1, false-accept/false-reject, latency/FPS, and
**detector-only vs detector + verification** ablation.

## Train YOLOv8 (transfer learning, Mosaic + photometric augmentation)
    python make_yolo_dataset.py 300           # synthetic defect generation -> datasets/pcb + data.yaml
    python train.py 30 960                    # -> weights_best.pt    (auto-uses GPU if available)
    python benchmark_onnx.py weights_best.pt  # ONNX export + CPU FPS
Then in the dashboard sidebar choose **YOLOv8 (trained weights)**.
For a REAL board, label 100-200 photos (Roboflow / LabelImg, YOLO format, classes in data.yaml) and put them in
datasets/pcb/images/{train,val} + labels/ - train.py is unchanged.

Google Colab (free GPU, ~5 min): upload this folder, `!pip install ultralytics`, run the three commands above
with `device=0` in train.py.

## Use on a real board
    python make_roi.py my_golden.jpg          # click a box around each component once -> data/config.json
Upload golden image + config.json in the dashboard sidebar (or replace data/golden_reference.png + config.json).
Tips: fixed camera/jig, diffused light, board fills frame, silkscreen/holes visible (ORB anchors).
Tune the three threshold sliders (tau_conf, tau_IoU, tau_delta) on a few known-good boards first.

## Tests
    PYTHONPATH=. python tests/test_yolo_path.py     # YOLO integration logic with a mock detector

## Real-board validation (public photos)
    python fetch_real_data.py      # downloads real photos of EB8.5_POWER boards (github.com/aryan-programmer/pcb-fault-detection),
                                   # crops 8 test boards + 1 golden board into data/real/
    python inspect_real.py         # -> out/real_results.csv, out/real_overlays.png
Golden reference = one complete board; 20 component regions annotated in data/real/config_real.json.
Dashboard: upload data/real/golden_reference.png + data/real/config_real.json in the sidebar, then any data/real/*_N.png.
Real-photo options in the config: photometric_norm (white-balance matching), local_refine (RANSAC affine absorbs
residual perspective), detect_extras=false (golden board has optional unpopulated pads).
