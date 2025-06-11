import os
import numpy as np
from pathlib import Path
import torch
from yolov5 import val  # Import from the YOLOv5 repo

# Load model
weights_path = './yolov5/runs/train/exp4/weights/best.pt'
device = 0 if torch.cuda.is_available() else 'cpu'

# Search ranges
conf_range = np.arange(0.3, 0.4, 0.05)
iou_range = np.arange(0.3, 0.4, 0.05)

best_map = 0
best_params = {}

for conf in conf_range:
    for iou in iou_range:
        print(f"Evaluating: conf={conf:.2f}, iou={iou:.2f}")
        metrics = val.run(
            data='yolov5/data/data.yaml',
            weights=weights_path,
            device=device,
            conf_thres=conf,
            iou_thres=iou,
            save_json=False,
            verbose=False
        )
        map50 = metrics[2]  # mAP@0.5
        print(type(map50))
        if isinstance(map50, tuple):
            map50 = map50[0]
        if map50 > best_map:
            best_map = map50
            best_params = {'conf': conf, 'iou': iou}

print(f"\n? Best mAP@0.5 = {best_map:.4f} at conf={best_params['conf']:.2f}, iou={best_params['iou']:.2f}")

