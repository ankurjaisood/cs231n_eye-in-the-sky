#!/bin/bash

# Script to run deepsort_cards.py on all videos in clips_416_square folder
# Outputs are organized in ALL_FILES_RUN{run_cnt} folders

# Note: We don't use set -e here because we want to continue processing even if one video fails

# Get absolute path of the current directory (where script is run from)
SCRIPT_DIR="$(pwd)"

# Configuration
CLIPS_DIR="git_datasets/clips_416_square"
DEEPSORT_DIR="deep_sort_pytorch"
DEEPSORT_SCRIPT="deepsort_cards.py"
CONFIG_DETECTION="configs/yolov5s_cards.yaml"
CONFIG_DEEPSORT="configs/deep_sort.yaml"
DATA_YAML="../yolov5/data_hilo.yaml"  # Adjust path as needed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== DeepSORT Cards Batch Processing ===${NC}"
echo -e "${BLUE}Script directory: $SCRIPT_DIR${NC}"

# Check if clips directory exists
if [ ! -d "$CLIPS_DIR" ]; then
    echo -e "${RED}Error: Clips directory '$CLIPS_DIR' not found!${NC}"
    echo -e "${YELLOW}Current directory contents:${NC}"
    ls -la
    exit 1
fi

echo -e "${GREEN}Found clips directory: $SCRIPT_DIR/$CLIPS_DIR${NC}"

# Check if deepsort script exists
if [ ! -f "$DEEPSORT_DIR/$DEEPSORT_SCRIPT" ]; then
    echo -e "${RED}Error: DeepSORT script '$DEEPSORT_SCRIPT' not found in '$DEEPSORT_DIR'!${NC}"
    echo -e "${YELLOW}Contents of $DEEPSORT_DIR:${NC}"
    ls -la "$DEEPSORT_DIR/"
    exit 1
fi

# Check if config files exist
if [ ! -f "$DEEPSORT_DIR/$CONFIG_DETECTION" ]; then
    echo -e "${RED}Error: Config file '$CONFIG_DETECTION' not found!${NC}"
    exit 1
fi

if [ ! -f "$DEEPSORT_DIR/$CONFIG_DEEPSORT" ]; then
    echo -e "${RED}Error: DeepSort config file '$CONFIG_DEEPSORT' not found!${NC}"
    exit 1
fi

if [ ! -f "$DEEPSORT_DIR/$DATA_YAML" ]; then
    echo -e "${RED}Error: Data YAML file '$DATA_YAML' not found!${NC}"
    exit 1
fi

# Find the next available run number
run_cnt=1
while [ -d "$SCRIPT_DIR/ALL_FILES_RUN${run_cnt}" ]; do
    ((run_cnt++))
done

# Create the main output directory (absolute path)
OUTPUT_DIR="$SCRIPT_DIR/ALL_FILES_RUN${run_cnt}"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Created output directory: $OUTPUT_DIR${NC}"

# Find all video files in the clips directory
echo -e "${YELLOW}Searching for videos in: $SCRIPT_DIR/$CLIPS_DIR${NC}"
video_files=($(find "$CLIPS_DIR" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.m4v" -o -iname "*.wmv" -o -iname "*.flv" -o -iname "*.webm" \) | sort))

echo -e "${BLUE}Total videos found: ${#video_files[@]}${NC}"

if [ ${#video_files[@]} -eq 0 ]; then
    echo -e "${RED}No video files found in '$CLIPS_DIR'!${NC}"
    echo -e "${YELLOW}Directory contents:${NC}"
    ls -la "$CLIPS_DIR"
    exit 1
fi

echo -e "${BLUE}Found ${#video_files[@]} video files to process:${NC}"
for i in "${!video_files[@]}"; do
    echo -e "  $((i+1)). $(basename "${video_files[i]}")"
done
echo

# Process each video file
successful=0
failed=0
failed_files=()

for i in "${!video_files[@]}"; do
    video_path="${video_files[i]}"
    video_name=$(basename "$video_path")
    video_basename="${video_name%.*}"  # Remove extension

    echo -e "${BLUE}=== Processing video $((i+1))/${#video_files[@]} ===${NC}"

    # Create subdirectory for this video's output (absolute path)
    video_output_dir="$OUTPUT_DIR/$video_basename"
    mkdir -p "$video_output_dir"

    echo -e "${YELLOW}Processing: $video_name${NC}"
    echo -e "  Input: $video_path"
    echo -e "  Output: $video_output_dir"

    # Run deepsort_cards.py (disable set -e for this section)
    set +e  # Allow errors without exiting

    cd "$SCRIPT_DIR/$DEEPSORT_DIR"
    echo "Current directory: $(pwd)"
    echo "Video path: $SCRIPT_DIR/$video_path"
    echo "Output path: $video_output_dir"

    python "$DEEPSORT_SCRIPT" \
        --VIDEO_PATH "$SCRIPT_DIR/$video_path" \
        --config_detection "$CONFIG_DETECTION" \
        --config_deepsort "$CONFIG_DEEPSORT" \
        --save_path "$video_output_dir" \
        --data_yaml "$DATA_YAML" \
        --max_frames_before_reset 5 \
        --conf_threshold 0.5

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully processed: $video_name${NC}"
        ((successful++))

        # Check if output files were actually created
        if [ -f "$video_output_dir/class_counts.json" ]; then
            echo -e "${GREEN}  ✓ class_counts.json created${NC}"
        else
            echo -e "${YELLOW}  ! class_counts.json not found${NC}"
        fi

        if [ -f "$video_output_dir/results.mp4" ]; then
            echo -e "${GREEN}  ✓ results.mp4 created${NC}"
        else
            echo -e "${YELLOW}  ! results.mp4 not found${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to process: $video_name (exit code: $exit_code)${NC}"
        failed_files+=("$video_name")
        ((failed++))
    fi

    cd "$SCRIPT_DIR"
    set -e  # Re-enable exit on error

    echo -e "${BLUE}Progress: $((successful + failed))/${#video_files[@]} completed${NC}"
    echo
done

# Create summary report (absolute path)
summary_file="$OUTPUT_DIR/batch_summary.txt"
cat > "$summary_file" << EOF
DeepSORT Cards Batch Processing Summary
======================================
Run ID: $run_cnt
Date: $(date)
Total files: ${#video_files[@]}
Successful: $successful
Failed: $failed

Processed Videos:
EOF

for video in "${video_files[@]}"; do
    echo "  - $(basename "$video")" >> "$summary_file"
done

if [ ${#failed_files[@]} -gt 0 ]; then
    echo "" >> "$summary_file"
    echo "Failed Videos:" >> "$summary_file"
    for failed_video in "${failed_files[@]}"; do
        echo "  - $failed_video" >> "$summary_file"
    done
fi

echo "Configuration Used:" >> "$summary_file"
echo "  DeepSORT Script: $DEEPSORT_SCRIPT" >> "$summary_file"
echo "  Config Detection: $CONFIG_DETECTION" >> "$summary_file"
echo "  Config DeepSort: $CONFIG_DEEPSORT" >> "$summary_file"
echo "  Data YAML: $DATA_YAML" >> "$summary_file"

# Final summary
echo -e "${BLUE}=== BATCH PROCESSING COMPLETE ===${NC}"
echo -e "${GREEN}Total files processed: ${#video_files[@]}${NC}"
echo -e "${GREEN}Successful: $successful${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}Failed: $failed${NC}"
    echo -e "${RED}Failed files: ${failed_files[*]}${NC}"
fi
echo -e "${BLUE}Results saved to: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Summary saved to: $summary_file${NC}"

# Optional: Create aggregated class counts
echo -e "${YELLOW}Creating aggregated class counts...${NC}"
aggregate_script="$OUTPUT_DIR/aggregate_counts.py"
cat > "$aggregate_script" << 'PYTHON_EOF'
#!/usr/bin/env python3
import json
import os
import glob

# Find all class_counts.json files
count_files = glob.glob("*/class_counts.json")
aggregated = {}

for count_file in count_files:
    video_name = os.path.dirname(count_file)
    try:
        with open(count_file, 'r') as f:
            counts = json.load(f)
        aggregated[video_name] = counts
        print(f"Loaded counts for: {video_name}")
    except Exception as e:
        print(f"Error loading {count_file}: {e}")

# Save aggregated results
with open("aggregated_class_counts.json", 'w') as f:
    json.dump(aggregated, f, indent=2)

print(f"\nAggregated counts saved to: aggregated_class_counts.json")
print(f"Total videos processed: {len(aggregated)}")
PYTHON_EOF

# Run the aggregation script from the output directory
cd "$OUTPUT_DIR"
python aggregate_counts.py
cd "$SCRIPT_DIR"

echo -e "${GREEN}Batch processing completed successfully!${NC}"