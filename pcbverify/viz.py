"""Intermediate-stage visuals for the 'pipeline view' (proves each stage really runs)."""
import cv2, numpy as np
from .align import preprocess


def keypoint_matches(test, ref, n=60):
    g1, g2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(g1, None); k2, d2 = orb.detectAndCompute(g2, None)
    ms = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = sorted([m for m, nn in (p for p in ms if len(p) == 2) if m.distance < 0.75 * nn.distance], key=lambda m: m.distance)
    img = cv2.drawMatches(test, k1, ref, k2, good[:n], None, matchColor=(60, 220, 60), flags=2)
    return img, len(k1), len(k2), len(good)


def diff_map(aligned, ref):
    a = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32); r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a *= r.mean() / max(a.mean(), 1e-6)
    d = np.clip(np.abs(a - r) * 3, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
