#!/usr/bin/env python3
"""
Steps:
  1) Walk through all files in input_dir whose extension is .png, .jpg, or .jpeg
  2) Resize each image to 256x256 pixels and save it as a PNG in output_img_dir/images
  3) Create a corresponding mask image of the same size (256x256) filled with zeros 
     (grayscale “L” mode) and save it in output_mask_dir/masks with the same base filename
"""

import os
from PIL import Image
import numpy as np

input_dir       = "/home/anksood/cs231n/cs231n_eye-in-the-sky/raw_blackjack_table_images"

output_image_dir = os.path.join(input_dir, "images")
output_mask_dir  = os.path.join(input_dir, "masks")

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Allowed image extensions (case‐insensitive)
valid_exts = {".png", ".jpg", ".jpeg"}

# Target size
TARGET_SIZE = (256, 256)

for fname in os.listdir(input_dir):
    base, ext = os.path.splitext(fname)
    if ext.lower() not in valid_exts:
        continue

    in_path = os.path.join(input_dir, fname)
    try:
        img = Image.open(in_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {fname}: could not open ({e})")
        continue

    output_filename = f"extra_{base}.png"

    # Resize to 256×256
    img_resized = img.resize(TARGET_SIZE, resample=Image.BILINEAR)

    # Create a zero mask (mode "L") of size 256×256
    mask_array = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
    mask_img = Image.fromarray(mask_array, mode="L")

    # Save as PNG in output_img_dir
    out_img_path = os.path.join(output_image_dir, output_filename)
    img_resized.save(out_img_path, format="PNG")

    # Save the mask with the same base name
    out_mask_path = os.path.join(output_mask_dir, output_filename)
    mask_img.save(out_mask_path, format="PNG")

    print(f"Processed {fname} → {output_filename}.png + mask {output_filename}")

print("Done.")
