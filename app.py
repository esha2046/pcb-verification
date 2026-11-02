"""Streamlit dashboard:  streamlit run app.py"""
import json, cv2, numpy as np, pandas as pd, streamlit as st
from pcbverify import inspect_board
from pcbverify.logger import log_result
from pcbverify.viz import keypoint_matches, diff_map

st.set_page_config(page_title="PCB Assembly Verification", layout="wide")
st.title("Vision-Based Assembly Verification & Missing Component Detection")
st.caption("RVCE · CCVR · Reference-guided, explainable inspection from a single golden board")

with st.sidebar:
    st.header("1 · Golden reference (one board)")
    ref_f = st.file_uploader("Reference image", type=["png", "jpg", "jpeg"], key="ref")
    cfg_f = st.file_uploader("Component map (JSON)", type=["json"], key="cfg")
    use_demo = st.checkbox("Use bundled demo reference", value=ref_f is None)
    st.header("2 · Detector")
    import os
    det_choice = st.radio("Backend", ["Template (1 reference, no training)", "YOLOv8 (trained weights)"])
    weights = st.text_input("YOLO weights path", "weights_best.pt") if det_choice.startswith("YOLO") else None
    st.header("3 · Thresholds")
    t_conf = st.slider("τ_conf (presence)", 0.2, 0.95, 0.75, 0.01)
    t_iou = st.slider("τ_IoU", 0.1, 0.95, 0.50, 0.01)
    t_delta = st.slider("τ_δ (centroid offset)", 0.002, 0.08, 0.02, 0.002)

if use_demo or not (ref_f and cfg_f):
    ref = cv2.imread("data/golden_reference.png"); cfg = json.load(open("data/config.json"))
else:
    ref = cv2.imdecode(np.frombuffer(ref_f.read(), np.uint8), 1); cfg = json.load(cfg_f)
cfg["thresholds"] = {"conf": t_conf, "iou": t_iou, "delta": t_delta}
detector = None
if weights:
    try:
        from pcbverify.detect import YoloDetector
        detector = YoloDetector(weights)
    except Exception as e:
        st.sidebar.error(f"Could not load YOLO weights: {e} - falling back to template detector")

st.header("Inspect a board")
mode = st.radio("Input", ["Image / webcam", "Video"], horizontal=True)
if mode == "Video":
    import tempfile
    from pcbverify.video import inspect_video
    vf = st.file_uploader("Upload a short video of the board (5-20 s, steady camera)", type=["mp4", "mov", "avi", "mkv"])
    every = st.slider("Inspect every N-th frame", 2, 30, 10)
    if vf:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(vf.read()); tmp.close()
        bar = st.progress(0.0)
        try:
            v = inspect_video(tmp.name, ref, cfg, every_n=every, progress=bar.progress, detector=detector)
        except RuntimeError as e:
            st.error(str(e)); st.stop()
        (st.success if v["status"] == "PASS" else st.error)(f"**{v['status']}** - video-level completeness {v['score']} %  ({v['frames_used']} frames used, {v['frames_skipped']} skipped)")
        if v["extra"]: st.warning("Extra / foreign object detected in most frames")
        a, b = st.columns([2, 1])
        a.image(cv2.cvtColor(v["worst_overlay"], cv2.COLOR_BGR2RGB), caption="Most informative frame (lowest score)", use_container_width=True)
        b.subheader("Score over time"); b.line_chart(pd.DataFrame(v["timeline"]).set_index("t")["score"])
        st.subheader("Per-component verdict (temporal vote)")
        st.dataframe(pd.DataFrame(v["final"]), use_container_width=True)
    st.stop()
c1, c2 = st.columns([1, 2])
with c1:
    up = st.file_uploader("Assembled board image (or webcam below)", type=["png", "jpg", "jpeg"])
    cam = st.camera_input("Webcam capture")
    demo = st.selectbox("…or pick a demo board", ["—", "test_good", "test_missing", "test_misplaced", "test_extra", "test_mixed"])
src = up or cam
img = (cv2.imdecode(np.frombuffer(src.getvalue(), np.uint8), 1) if src
       else cv2.imread(f"data/{demo}.png") if demo != "—" else None)

if img is not None:
    try:
        res = inspect_board(img, ref, cfg, detector=detector)
    except RuntimeError as e:
        st.error(f"{e}"); st.stop()
    s = res["summary"]
    (st.success if s["status"] == "PASS" else st.error)(f"**{s['status']}** — Assembly Completeness Score {s['score']} %")
    m = st.columns(5)
    for col, k in zip(m, ["expected", "correct", "missing", "misplaced", "extra"]):
        col.metric(k.title(), s[k])
    log_result(s, res["components"], getattr(src, "name", demo))
    with c2:
        tab1, tab2 = st.tabs(["Result", "Pipeline view"])
        tab1.image(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB), caption="green = correct · red = missing · amber = misplaced · blue = extra", use_container_width=True)
        with tab2:
            kp, n1, n2, ng = keypoint_matches(cv2.resize(img, (ref.shape[1], int(img.shape[0] * ref.shape[1] / img.shape[1]))), ref)
            st.markdown(f"**Stage 2 - ORB alignment:** {n1} / {n2} keypoints, {ng} good matches, **{s['align_inliers']} RANSAC inliers**")
            st.image(cv2.cvtColor(kp, cv2.COLOR_BGR2RGB), use_container_width=True)
            a1, a2 = st.columns(2)
            a1.image(cv2.cvtColor(res["aligned"], cv2.COLOR_BGR2RGB), caption="Aligned (homography-warped) test image", use_container_width=True)
            a2.image(cv2.cvtColor(diff_map(res["aligned"], ref), cv2.COLOR_BGR2RGB), caption="Difference vs golden reference", use_container_width=True)
    st.caption(f"Inference latency: {s['latency_ms']} ms  (~{1000 / s['latency_ms']:.1f} FPS)  ·  logged to out/inspection_log.csv")
    df = pd.DataFrame([{"component": r["name"], "class": r["cls"], "verdict": r["status"],
                        "confidence": round(r["conf"], 2), "IoU": round(r["iou"], 2),
                        "δ": None if r["delta"] is None else round(r["delta"], 4),
                        "SSIM": None if "ssim" not in r else round(r["ssim"], 2)} for r in res["components"]])
    st.subheader("Per-component explanation")
    st.dataframe(df.style.map(lambda v: {"correct": "background:#d6f5d6", "missing": "background:#f8c9c9", "misplaced": "background:#ffe3b0"}.get(v, ""), subset=["verdict"]), use_container_width=True)
    st.download_button("Download report (CSV)", df.to_csv(index=False), "inspection_report.csv")
else:
    st.info("Upload / capture a board image or choose a demo board.")
