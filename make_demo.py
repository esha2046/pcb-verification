"""Generates a synthetic PCB 'golden reference', its ROI config, and defective test boards
so the whole system can be demoed with no hardware/dataset."""
import json, cv2, numpy as np
rng = np.random.default_rng(7)
W, H = 1280, 900

def board():
    global rng
    rng = np.random.default_rng(7)  # same board every call
    img = np.full((H, W, 3), (60, 110, 40), np.uint8)             # green solder mask
    img = (img + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    for _ in range(60):                                            # copper traces
        x, y = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        cv2.line(img, (x, y), (x + int(rng.integers(-200, 200)), y), (80, 150, 90), 3)
        cv2.line(img, (x, y), (x, y + int(rng.integers(-150, 150))), (80, 150, 90), 3)
    for p in [(30, 30), (W - 30, 30), (30, H - 30), (W - 30, H - 30)]:   # mounting holes = good ORB anchors
        cv2.circle(img, p, 14, (200, 200, 200), -1); cv2.circle(img, p, 6, (30, 30, 30), -1)
    for i in range(40):                                            # silkscreen text for feature richness
        cv2.putText(img, f"R{i}", (int(rng.integers(60, W - 100)), int(rng.integers(60, H - 40))), 0, 0.5, (235, 235, 235), 1)
    return img

def part(img, c, dx=0, dy=0):
    x, y, w, h = c["roi"]; x += dx; y += dy; k = c["cls"]
    if k == "resistor":
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 45), -1)
        cv2.rectangle(img, (x, y), (x + 8, y + h), (190, 190, 200), -1); cv2.rectangle(img, (x + w - 8, y), (x + w, y + h), (190, 190, 200), -1)
    elif k == "capacitor":
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 90, 160), -1)
        cv2.putText(img, "C", (x + w // 2 - 6, y + h // 2 + 6), 0, 0.6, (230, 230, 230), 2)
    elif k == "ic":
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 25, 25), -1)
        for j in range(6): 
            cv2.rectangle(img, (x - 6, y + 8 + j * (h - 16) // 6), (x, y + 8 + j * (h - 16) // 6 + 5), (200, 200, 210), -1)
            cv2.rectangle(img, (x + w, y + 8 + j * (h - 16) // 6), (x + w + 6, y + 8 + j * (h - 16) // 6 + 5), (200, 200, 210), -1)
        cv2.circle(img, (x + 12, y + 12), 5, (200, 200, 200), -1)
    elif k == "led":
        cv2.circle(img, (x + w // 2, y + h // 2), w // 2, (40, 40, 230), -1)
    else:  # connector
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), -1)
        for j in range(5): cv2.rectangle(img, (x + 6 + j * (w - 12) // 5, y + 6), (x + 6 + j * (w - 12) // 5 + 10, y + h - 6), (30, 30, 30), -1)

layout = [("ic", 150, 110, 140, 100)] + [("ic", 560, 130, 120, 90)] + \
    [("resistor", 100 + 90 * i, 380, 60, 26) for i in range(6)] + \
    [("capacitor", 140 + 110 * i, 520, 50, 50) for i in range(5)] + \
    [("led", 900 + 60 * i, 180, 34, 34) for i in range(3)] + \
    [("connector", 880, 640, 230, 70)] + [("resistor", 700 + 80 * i, 400, 60, 26) for i in range(3)]
comps = [{"name": f"{k[:3].upper()}{i+1}", "cls": k, "roi": [x, y, w, h]} for i, (k, x, y, w, h) in enumerate(layout)]

def build(defects=None, warp=True, light=1.0, pose=None, noise=0.0):
    b = board(); defects = defects or {}
    for c in comps:
        if c["name"] in defects.get("missing", []): continue
        dx, dy = defects.get("shift", {}).get(c["name"], (0, 0)); part(b, c, dx, dy)
    for (x, y, w, h) in defects.get("extra", []):
        cv2.rectangle(b, (x, y), (x + w, y + h), (235, 235, 235), -1)
    b = np.clip(b * light, 0, 255).astype(np.uint8)
    if warp:  # simulate camera pose change
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[18, 12], [W - 25, 6], [W - 10, H - 20], [10, H - 8]])
        if pose is not None: dst = dst + pose
        b = cv2.warpPerspective(b, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=(40, 40, 40))
    if noise: b = np.clip(b + np.random.default_rng(1).normal(0, noise, b.shape), 0, 255).astype(np.uint8)
    return b

if __name__ == "__main__":
    json.dump({"width": W, "board": "DEMO-PCB-01", "thresholds": {"conf": 0.75, "iou": 0.5, "delta": 0.02},
               "components": comps}, open("data/config.json", "w"), indent=2)
    cv2.imwrite("data/golden_reference.png", build(warp=False))
    cv2.imwrite("data/test_good.png", build(light=0.9))
    cv2.imwrite("data/test_missing.png", build({"missing": ["CAP11"]}, light=1.1))
    cv2.imwrite("data/test_misplaced.png", build({"shift": {"RES6": (28, 9), "LED15": (-14, 12)}}))
    cv2.imwrite("data/test_extra.png", build({"extra": [(1000, 330, 50, 40)]}))
    cv2.imwrite("data/test_mixed.png", build({"missing": ["CAP13"], "shift": {"RES7": (30, 0)}, "extra": [(420, 700, 45, 45)]}, light=0.85))
    print("Demo data written to data/ -", len(comps), "components")
