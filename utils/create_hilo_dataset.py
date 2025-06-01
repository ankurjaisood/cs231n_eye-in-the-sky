#!/usr/bin/env python3
"""
Convert 52-class card dataset to 3-class Hi-Lo dataset for card counting.

Hi-Lo card counting system:
- Low (0): Cards 2-6
- None/Neutral (1): Cards 7-9
- High (2): Cards 10, J, Q, K, A
"""

# As a first step, copy the original dataset to a new directory
# The new directory should be called kaggle-image-detection-flat

import os
import shutil
from pathlib import Path


def create_class_mapping():
    """Create mapping from 52 card classes to 3 Hi-Lo classes"""

    # Original 52 classes from data.yaml
    card_classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
                   '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
                   '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
                   'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
                   'Qc', 'Qd', 'Qh', 'Qs']

    class_mapping = {}

    for i, card in enumerate(card_classes):
        # Extract card value (remove suit)
        if card.startswith('10'):
            value = '10'
        elif card[0] in 'AJQK':
            value = card[0]
        else:
            value = card[0]

        # Map to Hi-Lo classes
        if value in ['2', '3', '4', '5', '6']:
            hilo_class = 0  # Low
        elif value in ['7', '8', '9']:
            hilo_class = 1  # None/Neutral
        elif value in ['10', 'A', 'J', 'Q', 'K']:
            hilo_class = 2  # High
        else:
            raise ValueError(f"Unknown card value: {value} for card: {card}")

        class_mapping[i] = hilo_class
        print(f"Class {i:2d} ({card:2s}) -> Hi-Lo {hilo_class} ({'Low' if hilo_class==0 else 'None' if hilo_class==1 else 'High'})")

    return class_mapping


def update_label_file(label_path, class_mapping):
    """Update a single label file with new class mappings"""

    if not os.path.exists(label_path):
        return 0

    with open(label_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    conversions = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            print(f"Warning: Invalid label format in {label_path}: {line}")
            continue

        old_class = int(parts[0])
        x_center, y_center, width, height = parts[1:]

        if old_class in class_mapping:
            new_class = class_mapping[old_class]
            updated_lines.append(f"{new_class} {x_center} {y_center} {width} {height}\n")
            conversions += 1
        else:
            print(f"Warning: Unknown class {old_class} in {label_path}")
            # Keep original line if class not in mapping
            updated_lines.append(line + '\n')

    # Write updated file
    with open(label_path, 'w') as f:
        f.writelines(updated_lines)

    return conversions


def update_labels_directory(labels_dir, class_mapping):
    """Update all label files in a directory"""

    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"Directory not found: {labels_dir}")
        return 0

    total_conversions = 0
    file_count = 0

    print(f"\nUpdating labels in: {labels_dir}")

    for label_file in labels_path.glob("*.txt"):
        conversions = update_label_file(label_file, class_mapping)
        total_conversions += conversions
        file_count += 1

        if file_count % 100 == 0:
            print(f"  Processed {file_count} files, {total_conversions} conversions")

    print(f"  Completed: {file_count} files, {total_conversions} total conversions")
    return total_conversions


def create_hilo_data_yaml():
    """Create new data.yaml file for Hi-Lo classes"""

    hilo_yaml_content = """train: ../train/images
val: ../valid/images

nc: 3
names: ['low', 'none', 'high']
"""

    output_path = "datasets/kaggle-image-detection-flat/data_hilo.yaml"

    with open(output_path, 'w') as f:
        f.write(hilo_yaml_content)

    print(f"\nCreated Hi-Lo data.yaml: {output_path}")
    return output_path


def validate_conversion(labels_dir):
    """Validate that all classes are now in range 0-2"""

    labels_path = Path(labels_dir)
    class_counts = {0: 0, 1: 0, 2: 0}
    invalid_classes = set()

    for label_file in labels_path.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                if class_id in class_counts:
                    class_counts[class_id] += 1
                else:
                    invalid_classes.add(class_id)

    print(f"\nValidation results for {labels_dir}:")
    print(f"  Low (0):  {class_counts[0]:,} instances")
    print(f"  None (1): {class_counts[1]:,} instances")
    print(f"  High (2): {class_counts[2]:,} instances")

    if invalid_classes:
        print(f"  WARNING: Found invalid classes: {sorted(invalid_classes)}")
        return False
    else:
        print(f"  ✅ All classes valid (0-2)")
        return True


def main():
    """Main function to convert dataset"""

    print("=== Converting 52-Class Cards to 3-Class Hi-Lo Dataset ===")

    # Create class mapping
    print("\n1. Creating class mapping...")
    class_mapping = create_class_mapping()

    # Show mapping summary
    print(f"\nMapping Summary:")
    low_count = sum(1 for v in class_mapping.values() if v == 0)
    none_count = sum(1 for v in class_mapping.values() if v == 1)
    high_count = sum(1 for v in class_mapping.values() if v == 2)
    print(f"  Low (2-6):     {low_count} classes -> class 0")
    print(f"  None (7-9):    {none_count} classes -> class 1")
    print(f"  High (10-A):   {high_count} classes -> class 2")

    # Update training labels
    print("\n2. Updating training labels...")
    train_conversions = update_labels_directory(
        "datasets/kaggle-image-detection-flat/train/labels",
        class_mapping
    )

    # Update validation labels
    print("\n3. Updating validation labels...")
    val_conversions = update_labels_directory(
        "datasets/kaggle-image-detection-flat/valid/labels",
        class_mapping
    )

    # Create new data.yaml
    print("\n4. Creating Hi-Lo data.yaml...")
    hilo_yaml_path = create_hilo_data_yaml()

    # Validate conversion
    print("\n5. Validating conversion...")
    train_valid = validate_conversion("datasets/kaggle-image-detection-flat/train/labels")
    val_valid = validate_conversion("datasets/kaggle-image-detection-flat/valid/labels")

    # Summary
    print("\n=== Conversion Complete ===")
    print(f"Total conversions: {train_conversions + val_conversions:,}")
    print(f"Training set:      {train_conversions:,}")
    print(f"Validation set:    {val_conversions:,}")
    print(f"Validation:        {'✅ Success' if train_valid and val_valid else '❌ Issues found'}")
    print(f"New data file:     {hilo_yaml_path}")

    print(f"\nYou can now train YOLOv5 with:")
    print(f"python train.py --data {hilo_yaml_path} --weights yolov5s.pt --img 416 --epochs 50")


if __name__ == "__main__":
    main()