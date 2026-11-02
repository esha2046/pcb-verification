"""Builds a YOLO-format dataset (images + labels + data.yaml) by synthesising many variants of the golden board:
random removed / shifted parts, foreign objects, lighting and sensor noise.
This is exactly the 'synthetic defect generation' + augmentation from the risk table.
Images are in the *reference frame* (the pipeline feeds YOLO the ALIGNED image).

For a REAL board: put your own labelled photos (Roboflow / LabelImg, YOLO format) into the same folders -
train.py does not care where the images come from.
"""
import os, sys, random, shutil, cv2, numpy as np
import make_demo as md

CLASSES = ["resistor", "capacitor", "ic", "led", "connector", "foreign"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
rng = np.random.default_rng(11); random.seed(11)
root = "datasets/pcb"
shutil.rmtree(root, ignore_errors=True)
for sp in ("train", "val"):
    os.makedirs(f"{root}/images/{sp}"); os.makedirs(f"{root}/labels/{sp}")

def yolo_line(cls, box, W, H):
    x, y, w, h = box
    return f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"

for i in range(N):
    sp = "val" if i % 6 == 0 else "train"
    d = {"missing": [], "shift": {}}
    for c in md.comps:
        r = rng.random()
        if r < 0.12: d["missing"].append(c["name"])
        elif r < 0.30:
            w, h = c["roi"][2], c["roi"][3]
            d["shift"][c["name"]] = (int(rng.uniform(-.7, .7) * w), int(rng.uniform(-.5, .5) * h))
    img = md.build(d, warp=False, light=float(rng.uniform(0.75, 1.2)), noise=float(rng.uniform(0, 7)))
    labels = []
    for c in md.comps:
        if c["name"] in d["missing"]: continue
        dx, dy = d["shift"].get(c["name"], (0, 0)); x, y, w, h = c["roi"]
        labels.append(yolo_line(CLASSES.index(c["cls"]), (x + dx, y + dy, w, h), md.W, md.H))
    for _ in range(int(rng.integers(0, 3))):                     # foreign objects (random colour/size)
        w, h = int(rng.integers(30, 70)), int(rng.integers(30, 70))
        x, y = int(rng.integers(60, md.W - 130)), int(rng.integers(60, md.H - 130))
        col = tuple(int(v) for v in rng.integers(120, 256, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
        labels.append(yolo_line(5, (x, y, w, h), md.W, md.H))
    cv2.imwrite(f"{root}/images/{sp}/b{i:04d}.jpg", img)
    open(f"{root}/labels/{sp}/b{i:04d}.txt", "w").write("\n".join(labels))
open("data.yaml", "w").write(f"path: {os.path.abspath(root)}\ntrain: images/train\nval: images/val\nnames:\n" +
                             "".join(f"  {k}: {n}\n" for k, n in enumerate(CLASSES)))
print(f"dataset: {N} images -> {root}  (classes: {CLASSES})")
