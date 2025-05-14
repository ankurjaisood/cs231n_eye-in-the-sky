import cv2
import os
import random
import matplotlib.pyplot as plt
import math

NUMBER_IMAGES_TO_VISUALIZE = 5

# === Configuration ===
images_dir = 'datasets/kaggle-image-detection/train/images'
labels_dir = 'datasets/kaggle-image-detection/train/labels'
class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s',
               '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
               '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
               '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s',
               '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As',
               'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
               'Qc', 'Qd', 'Qh', 'Qs']

# === Collect all image files ===
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)
sample_files = image_files[:NUMBER_IMAGES_TO_VISUALIZE]

# === Layout: auto-calculate rows/columns ===
cols = min(NUMBER_IMAGES_TO_VISUALIZE, 5)
rows = math.ceil(NUMBER_IMAGES_TO_VISUALIZE / cols)
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.flatten() if NUMBER_IMAGES_TO_VISUALIZE > 1 else [axes]

for idx, image_name in enumerate(sample_files):
    image_path = os.path.join(images_dir, image_name)
    label_path = os.path.join(labels_dir, image_name.replace('.jpg', '.txt')) 
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # === Read label file ===
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                cls_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                cls_id = int(cls_id)

                # Convert from normalized to pixel coordinates
                x1 = int((x_center - bbox_width / 2) * w)
                y1 = int((y_center - bbox_height / 2) * h)
                x2 = int((x_center + bbox_width / 2) * w)
                y2 = int((y_center + bbox_height / 2) * h)

                # Draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_names[cls_id], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print(f"Label file not found: {label_path}")

    axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(image_name, fontsize=8)
    axes[idx].axis('off')

for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(top=0.9, hspace=0.4)
plt.show()