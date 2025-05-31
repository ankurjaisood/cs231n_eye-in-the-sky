import pickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

DATA_PATH = "/home/anksood/Downloads/archive (3)/scenes.pck"
OUT_PATH = "/home/anksood/cs231n/cs231n_eye-in-the-sky/datasets/kaggle-image-segmentation/"

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

for split in ("train", "valid", "test"):
    os.makedirs(os.path.join(OUT_PATH, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH, split, "masks"), exist_ok=True)

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
assert len(images) == len(masks), "Image and mask counts must match"
N = len(images)

print("Num samples:", len(images))
print(" - image[0] shape, dtype:", np.array(images[0]).shape, np.array(images[0]).dtype)
print(" - mask[0] shape, dtype:", np.array(masks[0]).shape, np.array(masks[0]).dtype)
print(" - info:", info)
print(" - Num labels:", len(info["labels"]))

# Validate mask pixel‐values
mins = [np.min(mask) for mask in masks]
maxs = [np.max(mask) for mask in masks]

global_min = int(np.min(mins))
global_max = int(np.max(maxs))
print(f"Mask pixel values range from {global_min} to {global_max}")
assert global_min == 0, "Mask pixel values must be non-negative"
assert global_max == len(info["labels"]), "Mask pixel values must be less than number of labels"

triples = [(i, images[i], masks[i]) for i in range(N)]
# Randomly shuffle indices so pairs stay together but order is random
perm = np.random.permutation(N)
shuffled = [triples[i] for i in perm]

# Train, Validation, Test split
n_train = int(N * TRAIN_RATIO)
n_vali  = int(N * VALID_RATIO)
# n_test = N - n_train - n_vali

train_split = shuffled[:n_train]
valid_split  = shuffled[n_train : n_train + n_vali]
test_split  = shuffled[n_train + n_vali :]

print("len train =", len(train_split))
print("len valid =", len(valid_split))
print("len test =", len(test_split))
assert len(train_split) + len(valid_split) + len(test_split) == N, "Split counts must match total samples"

def save_split(split_name, split_data):
    """
    split_name: "train", "valid", or "test"
    imgs, msks: lists of NumPy arrays
    Writes each pair into:
      OUT_PATH/{split_name}/images/{idx:05d}.png
      OUT_PATH/{split_name}/masks/{idx:05d}.png
    """
    img_folder  = os.path.join(OUT_PATH, split_name, "images")
    mask_folder = os.path.join(OUT_PATH, split_name, "masks")

    for orig_idx, img_arr, mask_arr in split_data:
        filename = f"{orig_idx:05d}.png"
        img_path  = os.path.join(img_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        # Convert NumPy → PIL and save
        img = Image.fromarray(img_arr)
        img.save(img_path)

        # Ensure mask_arr is uint8 (each pixel is an integer class‐ID)
        mask = Image.fromarray(mask_arr.astype(np.uint8))
        mask.save(mask_path)

    print(f"Saved {len(split_data)} samples under “{split_name}/”")

#save_split("train", train_split)
#save_split("valid",  valid_split)
#save_split("test",  test_split)
