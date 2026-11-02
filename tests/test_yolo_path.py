"""Tests the YOLO integration logic (assignment, misplaced, missing, extra) with a mock detector that
emits YOLO-style detections - no torch needed.  Run: PYTHONPATH=. python tests/test_yolo_path.py"""
import cv2, numpy as np, make_demo as md
from pcbverify import inspect_board, load_config

cfg = load_config("data/config.json"); ref = cv2.imread("data/golden_reference.png")

class MockYolo:
    """Pretends to be a trained YOLO: returns boxes for parts that are actually drawn (+ foreign blocks)."""
    def __init__(self, defects): self.d = defects
    def detect(self, img):
        out = []
        for c in md.comps:
            if c["name"] in self.d.get("missing", []): continue
            dx, dy = self.d.get("shift", {}).get(c["name"], (0, 0)); x, y, w, h = c["roi"]
            out.append({"name": None, "cls": c["cls"], "box": (x + dx, y + dy, w, h), "conf": 0.9})
        for (x, y, w, h) in self.d.get("extra", []):
            out.append({"name": None, "cls": "foreign", "box": (x, y, w, h), "conf": 0.8})
        return out

cases = {"good": {}, "missing": {"missing": ["CAP11"]}, "misplaced": {"shift": {"RES6": (28, 9), "LED15": (-14, 12)}},
         "extra": {"extra": [(420, 700, 45, 45)]}, "mixed": {"missing": ["CAP13"], "shift": {"RES7": (30, 0)}, "extra": [(420, 700, 45, 45)]}}
expect = {"good": (0, 0, 0), "missing": (1, 0, 0), "misplaced": (0, 2, 0), "extra": (0, 0, 1), "mixed": (1, 1, 1)}
ok = True
for k, d in cases.items():
    r = inspect_board(md.build(d), ref, cfg, detector=MockYolo(d))["summary"]
    got = (r["missing"], r["misplaced"], r["extra"]); good = got == expect[k]; ok &= good
    print(f"{k:10s} expected {expect[k]} got {got} -> {'OK' if good else 'FAIL'}   {r['status']} {r['score']}%")
print("ALL YOLO-PATH TESTS PASSED" if ok else "SOME TESTS FAILED")
