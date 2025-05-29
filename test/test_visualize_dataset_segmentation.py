import os
import random
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation"
IMG_DIR  = DATASET_DIR + "/images"
MASK_DIR = DATASET_DIR + "/masks"

image_files = [f for f in os.listdir(IMG_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise RuntimeError(f"No images found in {IMG_DIR!r}")

fname = random.choice(image_files)

img  = Image.open(os.path.join(IMG_DIR,  fname))
mask = Image.open(os.path.join(MASK_DIR, fname))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img)
ax1.set_title(f"Image: {fname}")
ax1.axis("off")

ax2.imshow(mask, cmap="nipy_spectral", interpolation="nearest")
ax2.set_title(f"Mask:  {fname}")
ax2.axis("off")

plt.tight_layout()
plt.show()
