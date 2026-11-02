"""Stages 4-6: reference verification, completeness score, PASS/FAIL + overlay."""
import json, math, time
import cv2
import numpy as np
from .align import preprocess, align_to_reference, photometric_normalize
from .detect import TemplateDetector, find_extras, assign_yolo, yolo_extras
from .metrics import iou, centroid_offset, completeness, ssim

GREEN, RED, AMBER, BLUE = (60, 180, 60), (40, 40, 220), (0, 170, 255), (220, 120, 40)


def load_config(path):
    return json.load(open(path))


def inspect_board(test_img, ref_img, cfg, detector=None, verify=True):
    """verify=False -> 'detector-only' baseline (presence by confidence, no IoU/centroid check) for ablation."""
    t0 = time.perf_counter()
    comps = cfg["components"]
    t = cfg.get("thresholds", {})
    t_conf, t_iou, t_delta = t.get("conf", 0.75), t.get("iou", 0.5), t.get("delta", 0.02)

    ref = preprocess(ref_img, cfg.get("width", ref_img.shape[1]))
    test = preprocess(test_img, ref.shape[1])
    aligned, H, inliers = align_to_reference(test, ref)
    diag = math.hypot(*ref.shape[:2][::-1])

    if cfg.get("photometric_norm", False):
        aligned = photometric_normalize(aligned, ref)
    detector = detector or TemplateDetector(ref, comps, mode=cfg.get("match_mode", "gray"), min_margin=cfg.get("min_margin", 0), blur=cfg.get("match_blur", 0.0))
    dets = detector.detect(aligned)
    is_tpl = isinstance(detector, TemplateDetector)
    if is_tpl:
        by_name, unassigned = {d["name"]: d for d in dets if d["name"]}, []
    else:
        by_name, unassigned = assign_yolo(dets, comps)

    # --- local refinement: residual perspective/scale error is modelled with a RANSAC affine fit on the
    #     confident matches, so each part is judged against where it SHOULD be on THIS board.
    exp = {c["name"]: list(c["roi"]) for c in comps}
    if is_tpl and cfg.get("local_refine", True):
        src, dst = [], []
        for c in comps:
            d = by_name.get(c["name"])
            if d and d["conf"] >= cfg.get("refine_conf", 0.6):
                x, y, w, h = c["roi"]; bx, by_, bw, bh = d["box"]
                src.append([x + w / 2, y + h / 2]); dst.append([bx + bw / 2, by_ + bh / 2])
        if len(src) >= 6:
            M, inl = cv2.estimateAffine2D(np.float32(src), np.float32(dst), ransacReprojThreshold=cfg.get("refine_px", 4.0))
            if M is not None:
                sc = math.sqrt(abs(np.linalg.det(M[:, :2])))
                for c in comps:
                    x, y, w, h = c["roi"]
                    cx, cy = M @ np.array([x + w / 2, y + h / 2, 1.0])
                    exp[c["name"]] = [int(cx - w * sc / 2), int(cy - h * sc / 2), int(w * sc), int(h * sc)]

    results = []
    for c in comps:
        d = by_name.get(c["name"])
        if d is None or d["conf"] < t_conf:
            results.append({**c, "roi": exp[c["name"]], "status": "missing", "conf": d["conf"] if d else 0, "iou": 0, "delta": None, "box": None})
            continue
        i, dl = iou(d["box"], exp[c["name"]]), centroid_offset(d["box"], exp[c["name"]], diag)
        ok = (i >= t_iou and dl <= t_delta) if verify else True
        rx, ry, rw, rh = c["roi"]
        ga = cv2.cvtColor(aligned[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(ref[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY)
        results.append({**c, "roi": exp[c["name"]], "status": "correct" if ok else "misplaced", "conf": d["conf"],
                        "iou": i, "delta": dl, "box": d["box"], "ssim": ssim(ga, gr)})

    extras = (find_extras(aligned, ref, comps) if is_tpl else yolo_extras(unassigned, comps)) if (verify and cfg.get("detect_extras", True)) else []
    n_ok = sum(r["status"] == "correct" for r in results)
    score = completeness(n_ok, len(comps))
    summary = {"expected": len(comps), "correct": n_ok,
               "missing": sum(r["status"] == "missing" for r in results),
               "misplaced": sum(r["status"] == "misplaced" for r in results),
               "extra": len(extras), "score": round(score, 1),
               "status": "PASS" if score == 100 and not extras else "FAIL",
               "align_inliers": inliers, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return {"summary": summary, "components": results, "extras": extras,
            "overlay": draw_overlay(aligned, results, extras, summary), "aligned": aligned}


def draw_overlay(img, results, extras, s):
    o = img.copy()
    for r in results:
        x, y, w, h = r["roi"]
        if r["status"] == "missing":
            cv2.rectangle(o, (x, y), (x + w, y + h), RED, 2)
            cv2.putText(o, f"{r['name']} MISSING", (x, y - 4), 0, 0.45, RED, 1, cv2.LINE_AA)
        else:
            col = GREEN if r["status"] == "correct" else AMBER
            bx, by, bw, bh = r["box"]
            cv2.rectangle(o, (bx, by), (bx + bw, by + bh), col, 2)
            if r["status"] == "misplaced":
                cv2.rectangle(o, (x, y), (x + w, y + h), RED, 1)
                cv2.putText(o, f"{r['name']} MISPLACED", (bx, by - 4), 0, 0.45, AMBER, 1, cv2.LINE_AA)
    for (x, y, w, h) in extras:
        cv2.rectangle(o, (x, y), (x + w, y + h), BLUE, 2)
        cv2.putText(o, "EXTRA", (x, y - 4), 0, 0.45, BLUE, 1, cv2.LINE_AA)
    banner = GREEN if s["status"] == "PASS" else RED
    cv2.rectangle(o, (0, 0), (o.shape[1], 34), banner, -1)
    cv2.putText(o, f"{s['status']}  |  Completeness {s['score']}%  |  missing {s['missing']}  misplaced {s['misplaced']}  extra {s['extra']}",
                (10, 23), 0, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return o
