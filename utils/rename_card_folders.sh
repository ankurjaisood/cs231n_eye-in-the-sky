#!/bin/bash

# Script to rename card classification folders to match data.yaml class names
# Usage: ./rename_card_folders.sh

DATASET_DIR="datasets/cards_train_classification"

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Directory $DATASET_DIR does not exist!"
    exit 1
fi

# Create associative array for folder name mapping
declare -A folder_map

# Number cards (2-10)
folder_map["two of clubs"]="2c"
folder_map["two of diamonds"]="2d"
folder_map["two of hearts"]="2h"
folder_map["two of spades"]="2s"
folder_map["three of clubs"]="3c"
folder_map["three of diamonds"]="3d"
folder_map["three of hearts"]="3h"
folder_map["three of spades"]="3s"
folder_map["four of clubs"]="4c"
folder_map["four of diamonds"]="4d"
folder_map["four of hearts"]="4h"
folder_map["four of spades"]="4s"
folder_map["five of clubs"]="5c"
folder_map["five of diamonds"]="5d"
folder_map["five of hearts"]="5h"
folder_map["five of spades"]="5s"
folder_map["six of clubs"]="6c"
folder_map["six of diamonds"]="6d"
folder_map["six of hearts"]="6h"
folder_map["six of spades"]="6s"
folder_map["seven of clubs"]="7c"
folder_map["seven of diamonds"]="7d"
folder_map["seven of hearts"]="7h"
folder_map["seven of spades"]="7s"
folder_map["eight of clubs"]="8c"
folder_map["eight of diamonds"]="8d"
folder_map["eight of hearts"]="8h"
folder_map["eight of spades"]="8s"
folder_map["nine of clubs"]="9c"
folder_map["nine of diamonds"]="9d"
folder_map["nine of hearts"]="9h"
folder_map["nine of spades"]="9s"
folder_map["ten of clubs"]="10c"
folder_map["ten of diamonds"]="10d"
folder_map["ten of hearts"]="10h"
folder_map["ten of spades"]="10s"

# Face cards
folder_map["jack of clubs"]="Jc"
folder_map["jack of diamonds"]="Jd"
folder_map["jack of hearts"]="Jh"
folder_map["jack of spades"]="Js"
folder_map["queen of clubs"]="Qc"
folder_map["queen of diamonds"]="Qd"
folder_map["queen of hearts"]="Qh"
folder_map["queen of spades"]="Qs"
folder_map["king of clubs"]="Kc"
folder_map["king of diamonds"]="Kd"
folder_map["king of hearts"]="Kh"
folder_map["king of spades"]="Ks"
folder_map["ace of clubs"]="Ac"
folder_map["ace of diamonds"]="Ad"
folder_map["ace of hearts"]="Ah"
folder_map["ace of spades"]="As"

echo "Starting folder renaming process..."
echo "Dataset directory: $DATASET_DIR"
echo

# Counter for renamed folders
renamed_count=0
skipped_count=0

# Iterate through all directories in the dataset folder
for folder in "$DATASET_DIR"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")

        # Check if folder name exists in our mapping
        if [[ -n "${folder_map[$folder_name]}" ]]; then
            new_name="${folder_map[$folder_name]}"
            new_path="$DATASET_DIR/$new_name"

            # Check if target folder already exists
            if [ -d "$new_path" ]; then
                echo "⚠️  Skipping '$folder_name' -> '$new_name' (target already exists)"
                ((skipped_count++))
            else
                echo "✅ Renaming: '$folder_name' -> '$new_name'"
                mv "$folder" "$new_path"
                if [ $? -eq 0 ]; then
                    ((renamed_count++))
                else
                    echo "❌ Error renaming '$folder_name'"
                fi
            fi
        else
            echo "❓ Unknown folder: '$folder_name' (no mapping found)"
            ((skipped_count++))
        fi
    fi
done

echo
echo "=== Summary ==="
echo "Folders renamed: $renamed_count"
echo "Folders skipped: $skipped_count"
echo

# List final directory contents
echo "=== Final directory contents ==="
ls -la "$DATASET_DIR"