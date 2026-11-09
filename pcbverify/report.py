"""PDF inspection reports (single image + video). Needs:  pip install fpdf2
make_report(...)        -> bytes of a PDF for one inspected board image
make_video_report(...)  -> bytes of a PDF for a video inspection
"""
import io, time
import cv2
from fpdf import FPDF

GREEN, RED, AMBER, BLUE, GREY = (46, 140, 60), (200, 45, 45), (230, 150, 20), (40, 100, 200), (110, 110, 110)
ROW = {"correct": (222, 245, 222), "missing": (250, 205, 205), "misplaced": (255, 232, 185)}


def _jpg(img_bgr, q=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return io.BytesIO(buf.tobytes())


class _PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10); self.set_text_color(*GREY)
        self.cell(0, 6, "RVCE - CCVR | PCB Assembly Verification Report", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200); self.line(10, 16, 200, 16); self.ln(4)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 8); self.set_text_color(*GREY)
        self.cell(0, 6, f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | reference-guided inspection (one golden board) | page {self.page_no()}", align="C")


def _banner(pdf, status, score, extra_line=""):
    col = GREEN if status == "PASS" else RED
    pdf.set_fill_color(*col); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, f"  {status}   |   Assembly Completeness Score: {score}%", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(60, 60, 60)
    if extra_line:
        pdf.set_font("Helvetica", "", 10); pdf.cell(0, 7, extra_line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _meta(pdf, meta):
    pdf.set_font("Helvetica", "", 10); pdf.set_text_color(60, 60, 60)
    for k, v in meta.items():
        pdf.set_font("Helvetica", "B", 10); pdf.cell(42, 6, f"{k}:"); pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, str(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _counts(pdf, counts):
    cols = [("Expected", GREY), ("Correct", GREEN), ("Missing", RED), ("Misplaced", AMBER), ("Extra", BLUE)]
    w = 190 / len(cols)
    for name, col in cols:
        pdf.set_text_color(*col); pdf.set_font("Helvetica", "B", 18); pdf.cell(w, 9, str(counts.get(name.lower(), 0)), align="C")
    pdf.ln(9)
    for name, col in cols:
        pdf.set_text_color(*GREY); pdf.set_font("Helvetica", "", 9); pdf.cell(w, 5, name, align="C")
    pdf.ln(8); pdf.set_text_color(60, 60, 60)


def _table(pdf, header, rows, widths, verdict_col):
    pdf.set_font("Helvetica", "B", 9); pdf.set_fill_color(55, 65, 80); pdf.set_text_color(255, 255, 255)
    for h, w in zip(header, widths): pdf.cell(w, 7, h, border=0, fill=True, align="C")
    pdf.ln(7); pdf.set_font("Helvetica", "", 9); pdf.set_text_color(40, 40, 40)
    for r in rows:
        if pdf.get_y() > 275:
            pdf.add_page(); pdf.set_font("Helvetica", "B", 9); pdf.set_fill_color(55, 65, 80); pdf.set_text_color(255, 255, 255)
            for h, w in zip(header, widths): pdf.cell(w, 7, h, fill=True, align="C")
            pdf.ln(7); pdf.set_font("Helvetica", "", 9); pdf.set_text_color(40, 40, 40)
        pdf.set_fill_color(*ROW.get(str(r[verdict_col]), (245, 245, 245)))
        for v, w in zip(r, widths): pdf.cell(w, 6, str(v), fill=True, align="C")
        pdf.ln(6)


def _fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def make_report(summary, components, overlay_bgr, diff_bgr=None, meta=None):
    pdf = _PDF(); pdf.set_auto_page_break(True, 15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, "Inspection Report", new_x="LMARGIN", new_y="NEXT")
    _meta(pdf, meta or {})
    _banner(pdf, summary["status"], summary["score"],
            f"Alignment inliers: {summary.get('align_inliers', '-')} | latency: {summary.get('latency_ms', '-')} ms")
    _counts(pdf, summary)
    # annotated result (+ difference map side by side if given)
    pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Annotated result  (green correct, red missing, amber misplaced, blue extra)", new_x="LMARGIN", new_y="NEXT")
    h, w = overlay_bgr.shape[:2]; y0 = pdf.get_y()
    if diff_bgr is not None:
        iw = 93; ih = min(iw * h / w, 120)
        pdf.image(_jpg(overlay_bgr), x=10, y=y0, w=iw, h=ih); pdf.image(_jpg(diff_bgr), x=107, y=y0, w=iw, h=ih)
        pdf.set_y(y0 + ih + 1); pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(*GREY)
        pdf.cell(96, 5, "Verdict overlay"); pdf.cell(0, 5, "Difference vs golden reference", new_x="LMARGIN", new_y="NEXT")
    else:
        iw = 120; ih = min(iw * h / w, 130)
        pdf.image(_jpg(overlay_bgr), x=45, y=y0, w=iw, h=ih); pdf.set_y(y0 + ih + 2)
    pdf.set_text_color(60, 60, 60)
    # issues list
    bad = [c for c in components if c["status"] != "correct"]
    pdf.ln(2); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, f"Issues found: {len(bad) + summary.get('extra', 0)}", new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 10)
    if not bad and not summary.get("extra"):
        pdf.cell(0, 6, "None - all expected components present and correctly placed.", new_x="LMARGIN", new_y="NEXT")
    for c in bad:
        pdf.cell(0, 6, f"- {c['name']} ({c['cls']}): {c['status'].upper()}  (confidence {c['conf']:.2f}, IoU {c['iou']:.2f})", new_x="LMARGIN", new_y="NEXT")
    if summary.get("extra"):
        pdf.cell(0, 6, f"- {summary['extra']} extra / foreign object(s) detected", new_x="LMARGIN", new_y="NEXT")
    # full table (problems first)
    pdf.add_page(); pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 9, "Per-component explanation", new_x="LMARGIN", new_y="NEXT")
    order = {"missing": 0, "misplaced": 1, "correct": 2}
    rows = [[c["name"], c["cls"], c["status"], _fmt(c["conf"]), _fmt(c["iou"]), _fmt(c["delta"], 4), _fmt(c.get("ssim"))]
            for c in sorted(components, key=lambda c: order.get(c["status"], 3))]
    _table(pdf, ["Component", "Class", "Verdict", "Confidence", "IoU", "Offset", "SSIM"], rows, [32, 32, 30, 30, 22, 22, 22], 2)
    pdf.ln(4); pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4, "Method: ORB/SIFT + RANSAC alignment to the golden board, template or YOLOv8 detection, IoU / centroid-offset / SSIM "
                         "verification, completeness = correct / expected x 100. PASS requires 100% and no extra objects.")
    return bytes(pdf.output())


def make_video_report(v, meta=None):
    """v = dict returned by pcbverify.video.inspect_video"""
    pdf = _PDF(); pdf.set_auto_page_break(True, 15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.set_text_color(30, 30, 30); pdf.cell(0, 9, "Video Inspection Report", new_x="LMARGIN", new_y="NEXT")
    _meta(pdf, meta or {})
    _banner(pdf, v["status"], v["score"], f"{v['frames_used']} frames used, {v['frames_skipped']} skipped (alignment failed) | extra object in most frames: {v['extra']}")
    # score-over-time chart drawn with plain lines
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(30, 30, 30); pdf.cell(0, 7, "Score over time", new_x="LMARGIN", new_y="NEXT")
    x0, y0, cw, ch = 20, pdf.get_y() + 2, 170, 45
    pdf.set_draw_color(180, 180, 180); pdf.rect(x0, y0, cw, ch)
    tl = v["timeline"]
    if len(tl) > 1:
        t0, t1 = tl[0]["t"], tl[-1]["t"]; span = (t1 - t0) or 1
        pts = [(x0 + (p["t"] - t0) / span * cw, y0 + ch - p["score"] / 100 * ch) for p in tl]
        pdf.set_draw_color(*BLUE); pdf.set_line_width(0.6)
        for a, b in zip(pts, pts[1:]): pdf.line(a[0], a[1], b[0], b[1])
        pdf.set_line_width(0.2)
    pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*GREY)
    pdf.text(x0 - 9, y0 + 3, "100%"); pdf.text(x0 - 6, y0 + ch, "0%"); pdf.text(x0, y0 + ch + 4, "start"); pdf.text(x0 + cw - 8, y0 + ch + 4, "end")
    pdf.set_y(y0 + ch + 8)
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(30, 30, 30); pdf.cell(0, 7, "Most informative frame (lowest score)", new_x="LMARGIN", new_y="NEXT")
    h, w = v["worst_overlay"].shape[:2]; iw = 110; ih = min(iw * h / w, 120); y = pdf.get_y()
    pdf.image(_jpg(v["worst_overlay"]), x=50, y=y, w=iw, h=ih); pdf.set_y(y + ih + 3)
    pdf.add_page(); pdf.set_font("Helvetica", "B", 13); pdf.cell(0, 9, "Per-component verdict (temporal vote)", new_x="LMARGIN", new_y="NEXT")
    order = {"missing": 0, "misplaced": 1, "correct": 2}
    rows = [[c["name"], c["cls"], c["verdict"], f"{c['frames_failed']}/{c['frames']}"] for c in sorted(v["final"], key=lambda c: order.get(c["verdict"], 3))]
    _table(pdf, ["Component", "Class", "Verdict", "Frames failed"], rows, [50, 45, 45, 45], 2)
    return bytes(pdf.output())