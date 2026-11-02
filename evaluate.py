"""Quantitative evaluation + ablation (slide 16 'Validation plan').

Generates N boards with RANDOM induced defects (removed / shifted / extra parts) under random
camera pose, lighting and sensor noise, runs the pipeline and reports:
  * per-defect-type precision / recall / F1
  * board-level false-accept & false-reject rates
  * latency / FPS on CPU
  * ablation: detector-only  vs  detector + reference verification
Usage:  python evaluate.py [--n 60] [--weights runs/.../best.pt]
"""
import argparse, json, time
import cv2, numpy as np, pandas as pd
import make_demo as md
from pcbverify import inspect_board, load_config

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--weights", default=None)
a = ap.parse_args()
cfg = load_config("data/config.json"); ref = cv2.imread("data/golden_reference.png")
rng = np.random.default_rng(2024)
names = [c["name"] for c in md.comps]; roi = {c["name"]: c["roi"] for c in md.comps}

det = None
if a.weights:
    from pcbverify.detect import YoloDetector
    det = YoloDetector(a.weights)

def random_case():
    d = {"missing": [], "shift": {}, "extra": []}
    if rng.random() < 0.25: return d, "good"
    for n in rng.choice(names, size=rng.integers(1, 4), replace=False):
        w, h = roi[n][2], roi[n][3]
        if rng.random() < 0.5: d["missing"].append(str(n))
        else: d["shift"][str(n)] = (int(rng.choice([-1, 1]) * rng.uniform(0.3, 0.55) * w), int(rng.choice([-1, 1]) * rng.uniform(0, 0.25) * h))
    if rng.random() < 0.35:
        x, y = int(rng.integers(100, 1100)), int(rng.integers(260, 360)) if rng.random() < .5 else int(rng.integers(740, 820))
        d["extra"].append((x, y, int(rng.integers(35, 55)), int(rng.integers(35, 55))))
    return d, "defective"

def truth(d):
    return {"missing": set(d["missing"]), "misplaced": set(d["shift"]), "extra": len(d["extra"]) > 0}

rows = []
for i in range(a.n):
    d, kind = random_case()
    pose = rng.uniform(-14, 14, (4, 2)).astype(np.float32)
    img = md.build(d, light=float(rng.uniform(0.8, 1.15)), pose=pose, noise=float(rng.uniform(0, 6)))
    t = truth(d)
    for mode, verify in (("detector_only", False), ("detector+verification", True)):
        try:
            r = inspect_board(img, ref, cfg, detector=det, verify=verify)
            pred = {"missing": {c["name"] for c in r["components"] if c["status"] == "missing"},
                    "misplaced": {c["name"] for c in r["components"] if c["status"] == "misplaced"},
                    "extra": len(r["extras"]) > 0}
            rows.append({"mode": mode, "board": i, "defective": kind == "defective", "pred_fail": r["summary"]["status"] == "FAIL",
                         "latency_ms": r["summary"]["latency_ms"], "t": t, "p": pred, "aligned": True})
        except RuntimeError:
            rows.append({"mode": mode, "board": i, "defective": kind == "defective", "pred_fail": True, "latency_ms": np.nan,
                         "t": t, "p": {"missing": set(), "misplaced": set(), "extra": False}, "aligned": False})
    print(f"\rboard {i+1}/{a.n}", end="")
print()

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

out = []
for mode in ("detector_only", "detector+verification"):
    R = [r for r in rows if r["mode"] == mode]
    rec = {"mode": mode}
    for k in ("missing", "misplaced"):
        tp = sum(len(r["t"][k] & r["p"][k]) for r in R); fp = sum(len(r["p"][k] - r["t"][k]) for r in R); fn = sum(len(r["t"][k] - r["p"][k]) for r in R)
        p, rc, f = prf(tp, fp, fn); rec.update({f"{k}_P": p, f"{k}_R": rc, f"{k}_F1": f})
    tp = sum(r["t"]["extra"] and r["p"]["extra"] for r in R); fp = sum((not r["t"]["extra"]) and r["p"]["extra"] for r in R); fn = sum(r["t"]["extra"] and not r["p"]["extra"] for r in R)
    p, rc, f = prf(tp, fp, fn); rec.update({"extra_P": p, "extra_R": rc, "extra_F1": f})
    bad = [r for r in R if r["defective"]]; good = [r for r in R if not r["defective"]]
    rec["false_accept_%"] = 100 * sum(not r["pred_fail"] for r in bad) / max(1, len(bad))
    rec["false_reject_%"] = 100 * sum(r["pred_fail"] for r in good) / max(1, len(good))
    rec["board_accuracy_%"] = 100 * sum(r["pred_fail"] == r["defective"] for r in R) / len(R)
    lat = np.nanmean([r["latency_ms"] for r in R]); rec["latency_ms"] = lat; rec["FPS"] = 1000 / lat
    rec["boards"] = len(R); rec["good/defective"] = f"{len(good)}/{len(bad)}"
    out.append(rec)
df = pd.DataFrame(out).set_index("mode").T
df.to_csv("out/evaluation_results.csv"); 
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(df.to_string()); print("\nSaved out/evaluation_results.csv")
