"""Downloads REAL PCB photos from the public GitHub repo aryan-programmer/pcb-fault-detection (test_images/),
crops the individual EB8.5_POWER boards and builds data/real/ (golden reference = a complete board).
The repo has no explicit licence, so images are fetched on your machine, not redistributed here.
Credit the source in your report/slides.   Usage: python fetch_real_data.py
"""
import os, urllib.request, cv2, numpy as np
BASE = "https://raw.githubusercontent.com/aryan-programmer/pcb-fault-detection/master/test_images/"
OUT = "data/real"; os.makedirs(OUT, exist_ok=True)
# file -> (x-range of right-hand board column, row ranges, tag)   [native pixel coordinates]
SPECS = {"mostly-ok.png": ((355, 735), [(0, 440), (450, 890), (905, 1345)], "mostly_ok"),
         "missing.PNG": ((388, 806), [(0, 470), (485, 975), (995, 1485)], "missing"),
         "missing-lots.png": ((390, 805), [(0, 425), (432, 890), (912, 1380)], "missing_lots")}
for f, ((x0, x1), rows, tag) in SPECS.items():
    path = f"{OUT}/_{f}"
    if not os.path.exists(path):
        print("downloading", f); urllib.request.urlretrieve(BASE + f, path)
    im = cv2.imread(path)
    for i, (a, b) in enumerate(rows):
        crop = im[a:b, x0:x1]
        if tag == "mostly_ok" and i == 2:      # complete board = golden reference (2x upscaled for ORB / template detail)
            cv2.imwrite(f"{OUT}/golden_reference.png", cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        else:
            cv2.imwrite(f"{OUT}/{tag}_{i + 1}.png", crop)
print("done ->", OUT, "(golden_reference.png + 8 test boards). Config: data/real/config_real.json")
