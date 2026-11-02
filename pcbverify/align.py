"""Stage 2: pre-processing + ORB keypoints + RANSAC homography alignment."""
import cv2
import numpy as np


def preprocess(img, width=1280):
    h, w = img.shape[:2]
    if w != width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def align_to_reference(test, ref, n_features=4000, ratio=0.75):
    """Warp `test` into the reference frame. Returns (aligned, H, n_inliers)."""
    g1 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(n_features)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        raise RuntimeError("No ORB features found - check image / lighting")
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 10:
        raise RuntimeError(f"Alignment failed: only {len(good)} good matches")
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    aligned = cv2.warpPerspective(test, H, (ref.shape[1], ref.shape[0]))
    return aligned, H, int(mask.sum())


def photometric_normalize(img, ref):
    """Reinhard-style colour transfer in LAB: match per-channel mean/std of `img` to `ref`.
    Removes white-balance / exposure differences between photos before comparison."""
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = a.sum(axis=2) > 30                       # ignore black warp borders
    for ch in range(3):
        ma, sa = a[..., ch][valid].mean(), a[..., ch][valid].std() + 1e-6
        mb, sb = b[..., ch].mean(), b[..., ch].std() + 1e-6
        a[..., ch] = (a[..., ch] - ma) * (sb / sa) + mb
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
