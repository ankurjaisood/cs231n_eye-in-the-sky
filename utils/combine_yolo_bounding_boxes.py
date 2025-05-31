#!/usr/bin/env python3
import os
import argparse

THIS_FILE = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(THIS_FILE)
INPUT_DIRS = [BASE_DIR + "/../datasets/kaggle-image-detection/test/labels",
              BASE_DIR + "/../datasets/kaggle-image-detection/train/labels",
              BASE_DIR + "/../datasets/kaggle-image-detection/valid/labels"]

def merge_corners_to_union(box_list):
    """
    Given a list of YOLO-format boxes (x_center, y_center, w, h), all normalized [0,1],
    compute the union box that tightly contains them all. Returns (x_center, y_center, w, h).
    """
    # Convert each (xc, yc, w, h) to (xmin, ymin, xmax, ymax)
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for xc, yc, w, h in box_list:
        xmin = xc - w / 2.0
        xmax = xc + w / 2.0
        ymin = yc - h / 2.0
        ymax = yc + h / 2.0
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

    # Union -> min of all xmin, min of all ymin, max of all xmax, max of all ymax
    xmin_u = min(xmins)
    ymin_u = min(ymins)
    xmax_u = max(xmaxs)
    ymax_u = max(ymaxs)

    # Convert back to YOLO (xc, yc, w, h)
    xc_u = (xmin_u + xmax_u) / 2.0
    yc_u = (ymin_u + ymax_u) / 2.0
    w_u = xmax_u - xmin_u
    h_u = ymax_u - ymin_u
    return xc_u, yc_u, w_u, h_u

def process_label_file(in_path, out_path):
    """
    Read a single YOLO-format .txt file at in_path, merge all corner-boxes per class,
    and write the merged boxes to out_path.
    """
    # dictionary: class_id (int) -> list of (xc, yc, w, h)
    boxes_by_class = {}

    with open(in_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                # skip malformed lines
                print(f"Skipping malformed line in {in_path}: {line}")
                continue

            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])

            if cls_id not in boxes_by_class:
                boxes_by_class[cls_id] = []
            boxes_by_class[cls_id].append((xc, yc, w, h))

    # Now merge each class’s boxes into one union-box
    merged_lines = []
    for cls_id, box_list in boxes_by_class.items():
        # If there’s only one box, the "union" is that same box.
        if len(box_list) == 1:
            xc_u, yc_u, w_u, h_u = box_list[0]
        else:
            xc_u, yc_u, w_u, h_u = merge_corners_to_union(box_list)

        # Clamp (optional) in case of tiny numerical overflow
        xc_u = max(0.0, min(1.0, xc_u))
        yc_u = max(0.0, min(1.0, yc_u))
        w_u  = max(0.0, min(1.0, w_u))
        h_u  = max(0.0, min(1.0, h_u))

        merged_lines.append(f"{cls_id} {xc_u:.6f} {yc_u:.6f} {w_u:.6f} {h_u:.6f}")

    with open(out_path, 'w') as f_out:
        for l in merged_lines:
            f_out.write(l + "\n")
            print(f"Writing merged box: {l} to {out_path}")

def main():

    for input_dir in INPUT_DIRS:
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
        output_dir = input_dir + "_merged"
        os.makedirs(output_dir, exist_ok=True)

        # Process every file ending in .txt
        for fname in os.listdir(input_dir):
            if not fname.lower().endswith(".txt"):
                print(f"Skipping non-txt file: {fname}")
                continue
            
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            process_label_file(in_path, out_path)
            #print(f"Processed {in_path} → {out_path}")

    print("Done. All label files have been merged.")

if __name__ == "__main__":
    main()
