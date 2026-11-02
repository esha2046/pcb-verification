"""Run the pipeline on all REAL test boards -> out/real_results.csv + out/real_overlays.png
python inspect_real.py"""
import cv2, glob, os, numpy as np, pandas as pd
from pcbverify import inspect_board, load_config
cfg = load_config("data/real/config_real.json"); ref = cv2.imread("data/real/golden_reference.png")
rows, tiles = [], []
for f in sorted(glob.glob("data/real/*_[123].png")):
    n = os.path.basename(f)[:-4]
    try:
        r = inspect_board(cv2.imread(f), ref, cfg)
    except RuntimeError as e:
        print(n, "alignment failed:", e); continue
    s = r["summary"]; flagged = [f"{c['name']}:{c['status']}" for c in r["components"] if c["status"] != "correct"]
    rows.append({"board": n, "status": s["status"], "score_%": s["score"], "missing": s["missing"], "misplaced": s["misplaced"],
                 "latency_ms": s["latency_ms"], "flagged": " ".join(flagged)})
    o = cv2.resize(r["overlay"], (380, 440)); cv2.putText(o, n, (4, 430), 0, 0.5, (0, 255, 255), 1); tiles.append(o)
os.makedirs("out", exist_ok=True)
df = pd.DataFrame(rows); df.to_csv("out/real_results.csv", index=False); print(df.drop(columns="flagged").to_string(index=False))
pad = [np.zeros((440, 380, 3), np.uint8)] * ((-len(tiles)) % 4)
tiles += pad; cv2.imwrite("out/real_overlays.png", np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
print("saved out/real_results.csv and out/real_overlays.png")
