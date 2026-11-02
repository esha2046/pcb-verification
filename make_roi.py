"""Define ROIs on YOUR real golden board:  python make_roi.py my_golden.jpg
Draw a box around each component -> ENTER, type name & class in terminal. ESC / empty box to finish."""
import sys, json, cv2
img = cv2.imread(sys.argv[1]); comps = []
while True:
    r = cv2.selectROI("Draw component box (ENTER=accept, ESC=finish)", img, showCrosshair=False)
    if r[2] == 0 or r[3] == 0: break
    name = input("name (e.g. R1): "); cls = input("class (resistor/capacitor/ic/led/connector): ")
    comps.append({"name": name, "cls": cls, "roi": [int(v) for v in r]})
    x, y, w, h = r; cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
json.dump({"width": img.shape[1], "board": sys.argv[1],
           "thresholds": {"conf": 0.55, "iou": 0.5, "delta": 0.02}, "components": comps}, open("data/config.json", "w"), indent=2)
print(len(comps), "components saved to data/config.json")
