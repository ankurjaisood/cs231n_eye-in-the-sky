import pickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_PATH = "/home/anksood/Downloads/archive (3)/scenes.pck"
OUT_PATH = "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/"
OUT_PATH_IMAGES = OUT_PATH + "images/"
OUT_PATH_MASKS = OUT_PATH + "masks/"

os.makedirs(OUT_PATH_IMAGES, exist_ok=True)
os.makedirs(OUT_PATH_MASKS, exist_ok=True)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

print(type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
elif isinstance(data, (list, tuple)):
    print("Length:", len(data), " ; example element type:", type(data[0]))

images = data["data"]
masks = data["gt"]
info = data["info"]

print("Num samples:", len(images))
print(" - image[0] shape, dtype:", np.array(images[0]).shape, np.array(images[0]).dtype)
print(" - mask[0] shape, dtype:", np.array(masks[0]).shape, np.array(masks[0]).dtype)
print(" - info:", info)
print(" - Num labels:", len(info["labels"]))


for i, (img_arr, mask_arr) in enumerate(zip(images, masks)):
    # save image
    img = Image.fromarray(img_arr)
    img.save(os.path.join(OUT_PATH_IMAGES, f"{i:05d}.png"))
    # save mask
    mask = Image.fromarray(mask_arr)
    mask.save(os.path.join(OUT_PATH_MASKS, f"{i:05d}.png"))

print(f"Exported {i+1} images → {OUT_PATH_IMAGES}")
print(f"Exported {i+1} masks  → {OUT_PATH_MASKS}")