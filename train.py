"""Fine-tune YOLOv8n (transfer learning from COCO weights) with Mosaic + photometric augmentation.
python train.py [epochs] [imgsz]
"""
import sys, shutil
import torch
from ultralytics import YOLO
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 25
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960
model = YOLO("yolov8n.pt")                       # pretrained COCO weights -> transfer learning
model.train(data="data.yaml", epochs=epochs, imgsz=imgsz, batch=8, device=(0 if torch.cuda.is_available() else "cpu"), workers=2,
            mosaic=1.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=0, fliplr=0.0,   # photometric + mosaic; no flips (layout matters)
            project="runs", name="pcb", exist_ok=True, patience=15, plots=True, verbose=False)
shutil.copy(model.trainer.best, "weights_best.pt")
print("saved weights_best.pt")
