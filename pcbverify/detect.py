"""Stage 3: component detection. Two interchangeable backends.

* TemplateDetector - works with ONE golden board, no training (default).
* YoloDetector     - plug in fine-tuned YOLOv8 weights once you have a dataset.
Both return: [{"name","cls","box":(x,y,w,h),"conf"}]
"""
import cv2
import numpy as np


class TemplateDetector:
    """Per-component template matching inside a search window around the expected ROI.
    Peaks that land on a *neighbouring* component's expected region are suppressed, so a removed
    part cannot 'steal' its neighbour's appearance (important for rows of identical resistors)."""
    def __init__(self, ref_img, components, margin=0.6, mode="gray", min_margin=0, blur=0.0):
        self.comps, self.margin, self.mode, self.min_margin, self.blur = components, margin, mode, min_margin, blur
        self.ref_gray = self._feat(ref_img)
        self.H, self.W = self.ref_gray.shape

    def _feat(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.mode == "edge":
            g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
            g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
            m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
            return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.blur:
            g = cv2.GaussianBlur(g, (0, 0), self.blur)
        return g

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)

    def detect(self, aligned):
        g = self._feat(aligned)
        out = []
        for c in self.comps:
            x, y, w, h = c["roi"]
            tpl = self.ref_gray[y:y + h, x:x + w]
            mx, my = max(int(w * self.margin) + 4, self.min_margin), max(int(h * self.margin) + 4, self.min_margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(self.W, x + w + mx), min(self.H, y + h + my)
            win = g[y0:y1, x0:x1]
            if win.shape[0] < h or win.shape[1] < w:
                continue
            res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
            others = [o["roi"] for o in self.comps if o is not c]
            best = (0.0, (0, 0))
            for _ in range(8):
                _, conf, _, loc = cv2.minMaxLoc(res)
                if conf <= 0:
                    break
                box = (x0 + loc[0], y0 + loc[1], w, h)
                if any(self._iou(box, o) > 0.2 for o in others):      # belongs to a neighbour
                    res[max(0, loc[1] - 3):loc[1] + 4, max(0, loc[0] - 3):loc[0] + 4] = -1
                    continue
                best = (conf, loc); break
            out.append({"name": c["name"], "cls": c["cls"],
                        "box": (x0 + best[1][0], y0 + best[1][1], w, h), "conf": float(max(best[0], 0))})
        return out


class YoloDetector:
    """Optional: pip install ultralytics, then pass weights path in config."""
    def __init__(self, weights, imgsz=1280):
        from ultralytics import YOLO
        self.model, self.imgsz = YOLO(weights), imgsz

    def detect(self, aligned):
        r = self.model.predict(aligned, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = b
            out.append({"name": None, "cls": r.names[int(k)],
                        "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), "conf": float(c)})
        return out


def find_extras(aligned, ref, components, thresh=55, min_area=450, pad=6, border=45):
    """Foreign / extra objects: difference blobs lying outside every expected ROI.
    Gain-compensated (lighting) and border-masked (warp artefacts)."""
    a = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a *= (r.mean() / max(a.mean(), 1e-6))
    d = np.abs(cv2.GaussianBlur(a, (5, 5), 0) - cv2.GaussianBlur(r, (5, 5), 0)).astype(np.uint8)
    _, m = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
    m[:border], m[-border:], m[:, :border], m[:, -border:] = 0, 0, 0, 0
    for c in components:  # mask expected zones (+search margin) so moved parts are not "extra"
        x, y, w, h = c["roi"]
        mx, my = int(w * 0.6) + pad, int(h * 0.6) + pad
        m[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][4] >= min_area]


def assign_yolo(dets, components, max_dist=2.5):
    """Greedy one-to-one assignment of YOLO detections to expected components.
    A detection may be assigned if same class and within max_dist * ROI-size of the expected centre
    (so a shifted part is reported as MISPLACED, not missing). Returns {name: det}."""
    import math
    pairs = []
    for c in components:
        x, y, w, h = c["roi"]; cx, cy = x + w / 2, y + h / 2; lim = max_dist * max(w, h)
        for j, d in enumerate(dets):
            if d["cls"] != c["cls"]:
                continue
            bx, by, bw, bh = d["box"]
            dist = math.dist((cx, cy), (bx + bw / 2, by + bh / 2))
            if dist <= lim:
                pairs.append((dist, c["name"], j))
    pairs.sort()
    used_c, used_d, out = set(), set(), {}
    for dist, name, j in pairs:
        if name in used_c or j in used_d:
            continue
        used_c.add(name); used_d.add(j); out[name] = dets[j]
    return out, [d for j, d in enumerate(dets) if j not in used_d]


def yolo_extras(unassigned, components, min_conf=0.5):
    """Detections not explained by any expected component = extra / foreign parts."""
    return [tuple(d["box"]) for d in unassigned if d["conf"] >= min_conf]
