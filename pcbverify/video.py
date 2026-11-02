"""Video inspection: sample frames, inspect each, aggregate with temporal voting
so one blurry/glared frame cannot cause a false alarm."""
import cv2
from collections import defaultdict
from .pipeline import inspect_board


def inspect_video(path, ref, cfg, every_n=10, max_frames=60, vote=0.5, progress=None, detector=None):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, idx = [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            try:
                r = inspect_board(f, ref, cfg, detector=detector)
                frames.append({"frame": idx, "t": idx / fps, "res": r})
            except RuntimeError:          # alignment failed (blur / board out of view) -> skip frame
                frames.append({"frame": idx, "t": idx / fps, "res": None})
            if progress: progress(min(1.0, len(frames) / max_frames))
        idx += 1
    cap.release()

    valid = [f for f in frames if f["res"]]
    if not valid:
        raise RuntimeError("Board could not be aligned in any sampled frame")
    # temporal voting per component
    votes = defaultdict(lambda: defaultdict(int))
    for f in valid:
        for c in f["res"]["components"]:
            votes[c["name"]][c["status"]] += 1
        votes["__extra__"]["extra" if f["res"]["extras"] else "none"] += 1
    n, final = len(valid), []
    for c in cfg["components"]:
        v = votes[c["name"]]
        bad = {k: v[k] for k in ("missing", "misplaced") if v[k]}
        worst = max(bad, key=bad.get) if bad else None
        status = worst if worst and bad[worst] / n >= vote else "correct"
        final.append({"name": c["name"], "cls": c["cls"], "verdict": status,
                      "frames_failed": sum(bad.values()), "frames": n})
    extra_flag = votes["__extra__"]["extra"] / n >= vote
    # timeline of score, and the most informative frame (lowest score) for the report
    timeline = [{"t": round(f["t"], 2), "score": f["res"]["summary"]["score"],
                 "status": f["res"]["summary"]["status"]} for f in valid]
    worst_frame = min(valid, key=lambda f: f["res"]["summary"]["score"])
    ok = sum(c["verdict"] == "correct" for c in final)
    score = 100.0 * ok / len(final)
    return {"final": final, "extra": extra_flag, "score": round(score, 1),
            "status": "PASS" if score == 100 and not extra_flag else "FAIL",
            "timeline": timeline, "worst_overlay": worst_frame["res"]["overlay"],
            "frames_used": n, "frames_skipped": len(frames) - n}
