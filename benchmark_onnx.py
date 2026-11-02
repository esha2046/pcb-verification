"""Export trained YOLOv8 to ONNX and benchmark CPU latency / FPS (supports the 'edge-friendly' claim).
python benchmark_onnx.py weights_best.pt"""
import sys, time, cv2, numpy as np
from ultralytics import YOLO
w = sys.argv[1] if len(sys.argv) > 1 else "weights_best.pt"
imgsz = 960
onnx = YOLO(w).export(format="onnx", imgsz=imgsz, simplify=False)
img = cv2.imread("data/golden_reference.png")
rows = []
for name, path in (("PyTorch (.pt)", w), ("ONNX", onnx)):
    m = YOLO(path, task="detect")
    for _ in range(3): m.predict(img, imgsz=imgsz, device="cpu", verbose=False)       # warm-up
    t = time.perf_counter(); n = 15
    for _ in range(n): m.predict(img, imgsz=imgsz, device="cpu", verbose=False)
    ms = (time.perf_counter() - t) / n * 1000; rows.append((name, ms, 1000 / ms))
print(f"\n{'model':16s} {'ms/frame':>10s} {'FPS':>8s}  (CPU, imgsz={imgsz})")
for n_, ms, fps in rows: print(f"{n_:16s} {ms:10.1f} {fps:8.1f}")
open("out/onnx_benchmark.txt", "w").write("\n".join(f"{n_},{ms:.1f},{fps:.1f}" for n_, ms, fps in rows))
