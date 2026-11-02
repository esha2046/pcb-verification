"""python inspect_cli.py data/test_mixed.png [--ref data/golden_reference.png --cfg data/config.json]"""
import argparse, json, cv2, os
from pcbverify import inspect_board, load_config
ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("--ref", default="data/golden_reference.png"); ap.add_argument("--cfg", default="data/config.json")
a = ap.parse_args()
if a.image.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    from pcbverify.video import inspect_video
    v = inspect_video(a.image, cv2.imread(a.ref), load_config(a.cfg))
    os.makedirs("out", exist_ok=True); cv2.imwrite("out/video_worst_frame.png", v["worst_overlay"])
    print(f"VIDEO VERDICT: {v['status']}  score={v['score']}%  extra={v['extra']}  frames={v['frames_used']} (skipped {v['frames_skipped']})")
    for c in v["final"]:
        if c["verdict"] != "correct": print(f"  {c['name']:8s} {c['verdict']:10s} failed in {c['frames_failed']}/{c['frames']} frames")
    print("timeline:", [(t['t'], t['score']) for t in v['timeline'][::3]]); raise SystemExit
res = inspect_board(cv2.imread(a.image), cv2.imread(a.ref), load_config(a.cfg))
os.makedirs("out", exist_ok=True)
out = "out/" + os.path.splitext(os.path.basename(a.image))[0] + "_result.png"
cv2.imwrite(out, res["overlay"])
print(json.dumps(res["summary"], indent=2))
for r in res["components"]:
    if r["status"] != "correct": print(f"  {r['name']:8s} {r['status']:10s} conf={r['conf']:.2f} iou={r['iou']:.2f}")
print("overlay ->", out)
