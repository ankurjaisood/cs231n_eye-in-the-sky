#!/usr/bin/env python3
"""
Hi-Lo groups:
   - Low cards  (2,3,4,5,6)       → group “Low”
   - Neutral cards (7,8,9)        → group “Neutral”
   - High cards (10,J,Q,K,A)      → group “High”

Originally:
id2label = {
    0: "background",
    1:  "A_C",  2: "2_C",  3: "3_C",  4: "4_C",  5: "5_C",  6: "6_C",  7: "7_C",  8: "8_C",  9: "9_C", 10: "10_C", 11: "J_C",  12:"Q_C", 13: "K_C",
    14: "A_S", 15: "2_S", 16: "3_S", 17: "4_S", 18: "5_S", 19: "6_S", 20: "7_S", 21: "8_S", 22: "9_S", 23: "10_S", 24: "J_S", 25: "Q_S", 26: "K_S",
    27: "A_H", 28: "2_H", 29: "3_H", 30: "4_H", 31: "5_H", 32: "6_H", 33: "7_H", 34: "8_H", 35: "9_H", 36: "10_H", 37: "J_H", 38: "Q_H", 39: "K_H",
    40: "A_D", 41: "2_D", 42: "3_D", 43: "4_D", 44: "5_D", 45: "6_D", 46: "7_D", 47: "8_D", 48: "9_D", 49: "10_D", 50: "J_D", 51: "Q_D", 52: "K_D",
}

Now:

id2label_suits_category = {
    0:  "background",
    1:  "C_Low",  2: "C_None", 3:  "C_High",
    4:  "H_Low",  5: "H_None", 6:  "H_High",
    7:  "S_Low",  8: "S_None", 9:  "S_High",
    10: "D_Low", 11: "D_None", 12: "D_High",
}

id2label_category = {
    0:  "background",
    1:  "Low", 
    2:  "None", 
    3:  "High",
}

This script will:
  1) Walk through all files in input_dir (“.png”, “.jpg”, “.jpeg”)
  2) Load each mask via PIL as an (H×W) numpy array of ints [0..52]
  3) Build a remapping lookup table old→new of length 53
  4) Apply that mapping to each pixel, creating a new (H×W) array with values in [0..12]
  5) Save the new mask (as “L” mode PNG) into output_dir, preserving filename
"""

import os
import argparse
from PIL import Image
import numpy as np

def remap_53_to_4(mask53: np.ndarray) -> np.ndarray:
    """
    Given a 2D numpy array `mask53` whose pixel‐values are in {0, 1, …, 52},
    return a new array of exactly the same shape whose values are in {0,1,2,3}:
      0 - background
      1 - Low    (original ranks 2-6)
      2 - None   (original ranks 7-9)
      3 - High   (original ranks 10,J=11,Q=12,K=13,A=1)
    """
    # Prepare an empty output mask (uint8 is enough, since only 0..3).
    mask4 = np.zeros_like(mask53, dtype=np.uint8)

    # We only need to iterate over all unique values present in mask53.
    unique_vals = np.unique(mask53)
    for orig_val in unique_vals:
        if orig_val == 0:
            # background stays 0
            continue

        # Compute “rank_index” in [1..13]:
        #   orig_val = 1 → A (rank_index=1)
        #   orig_val = 2 → 2 (rank_index=2)
        #   ...
        #   orig_val = 13 → K (rank_index=13)
        #   orig_val = 14 → A (rank_index=1), etc.
        rank_index = ((int(orig_val) - 1) % 13) + 1

        # Decide which new label to assign:
        if 2 <= rank_index <= 6:
            new_label = 1  # Low
        elif 7 <= rank_index <= 9:
            new_label = 2  # None
        else:
            # ranks {10,11,12,13,1} → High
            new_label = 3

        # Apply it to all pixels that had orig_val
        mask4[mask53 == orig_val] = new_label

    return mask4

def remap_53_to_13(mask53: np.ndarray) -> np.ndarray:
    """
    Given a 2D numpy array `mask53` with values in {0,1,…,52}, return a new array of the same shape
    whose values are in {0,1,…,12}, where:
      0  - background
      1  - C_Low
      2  - C_None
      3  - C_High
      4  - S_Low
      5  - S_None
      6  - S_High
      7  - H_Low
      8  - H_None
      9  - H_High
      10 - D_Low
      11 - D_None
      12 - D_High

    We compute, for each nonzero pixel:
        suit_index = (orig_id - 1) // 13      -> 0=Clubs, 1=Spades, 2=Hearts, 3=Diamonds
        rank_index = ((orig_id - 1) % 13) + 1 -> 1=Ace, 2=2, … 13=King

    Then we assign “Low” if rank_index  {2,3,4,5,6},
                   “None” if rank_index {7,8,9},
                   “High” if rank_index {1,10,11,12,13}).

    Finally, the new class id (1..12) is:
        new_id = suit_index*3 + group_id,
      where group_id = 1 for “Low”, 2 for “None”, 3 for “High”.
    (Thus Clubs occupy 1-3, Spades 4-6, Hearts 7-9, Diamonds 10-12.)
    """
    # Allocate output array (uint8 is enough since values 0..13)
    mask13 = np.zeros_like(mask53, dtype=np.uint8)

    # Only visit each unique value once
    unique_vals = np.unique(mask53)
    for orig_val in unique_vals:
        if orig_val == 0:
            # background → 0
            continue
    
        # suit_index = 0..3
        suit_index = (orig_val - 1) // 13
        # rank_index = 1..13
        rank_index = ((orig_val - 1) % 13) + 1

        # decide Low / None / High
        if rank_index in {2, 3, 4, 5, 6}:
            group_id = 1
        elif rank_index in {7, 8, 9}:
            group_id = 2
        else:
            # {1 (Ace),10,11(J),12(Q),13(K)} → High
            group_id = 3

        new_id = suit_index * 3 + group_id

        # Assign that rank to every pixel that was orig_val
        mask13[mask53 == orig_val] = new_id

    return mask13

def remap_one_mask(in_path: str, dst13_dir: str, dst4_dir: str, fname: str):
    """
    - Loads a grayscale (L-mode) mask image whose pixel‐values are in [0..52].
    - Applies remap_table so that each old value → new 0..12.
    - Saves the result as a new L-mode PNG at out_path.
    """
    # Load as 'L' (8-bit pixels, 0..255)
    img = Image.open(in_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # Make sure there are no stray values >52
    if arr.max() > 52:
        print(f"Warning: {in_path} has a max pixel value of {arr.max()}. Clipping to 52.")
        arr = np.clip(arr, 0, 52)

    remapped_13 = remap_53_to_13(arr)
    remapped_4 = remap_53_to_4(arr)

    if remapped_13.max() > 12 or remapped_4.max() > 3:
        raise ValueError(f"Remapping error: max values are {remapped_13.max()} (13-mask) and {remapped_4.max()} (4-mask).")

    out_img = Image.fromarray(remapped_13, mode="L")
    out_img.save(os.path.join(dst13_dir, fname))
    out_img = Image.fromarray(remapped_4, mode="L")
    out_img.save(os.path.join(dst4_dir, fname))

def main():
    input_dir = "./datasets/kaggle-image-segmentation/valid/masks/"
    output_dir = "./datasets/kaggle-image-segmentation/valid/"
    exts = ["png"]
    
    dst13 = os.path.join(output_dir, "masks_13")
    dst4  = os.path.join(output_dir, "masks_4")
    os.makedirs(dst13, exist_ok=True)
    os.makedirs(dst4,  exist_ok=True)

    exts_lower = {e.lower().lstrip(".") for e in exts}
    all_files = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and f.lower().split(".")[-1] in exts_lower
    ]

    if len(all_files) == 0:
        print(f"No mask files with extensions {exts} found in {input_dir}. Exiting.")
        return

    print(f"Found {len(all_files)} masks to process in {input_dir}.")
    for fname in all_files:
        in_path = os.path.join(input_dir, fname)
        remap_one_mask(in_path, dst13, dst4, fname)
        print(f"{fname} → masks_13/{fname}, masks_4/{fname}")

    print("\nDone.")
    print(f"  13-class masks are in: {dst13}")
    print(f"   4-class masks are in: {dst4}")

if __name__ == "__main__":
    main()
