#!/usr/bin/env python3
"""
Aggregate SegFormer results into format compatible with evaluate_predictions.py
"""

import json
import os
import argparse
import sys
from pathlib import Path

def aggregate_segformer_results(input_folder, output_file):
    """
    Aggregate SegFormer JSON results into aggregated_class_counts.json format

    Args:
        input_folder: Path to folder containing SegFormer JSON result files
        output_file: Path to output aggregated JSON file
    """

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        sys.exit(1)

    aggregated_results = {}
    processed_files = 0
    skipped_files = []

    print(f"Processing SegFormer results from: {input_folder}")
    print("-" * 60)

    # Get all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    if not json_files:
        print(f"Error: No JSON files found in {input_folder}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files to process")

    for filename in sorted(json_files):
        file_path = os.path.join(input_folder, filename)

        try:
            # Load the SegFormer result file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract video name from filename
            # Convert "1_416px_10fps.json" -> "1.mp4"
            video_name = filename.replace('_416px_10fps.json', '.mp4')

            # Get counts from the aggregated Low/None/High arrays
            low_count = len(data.get('Low', []))
            none_count = len(data.get('None', []))
            high_count = len(data.get('High', []))

            # Store in the same format as aggregated_class_counts.json
            aggregated_results[video_name] = {
                "low": low_count,
                "none": none_count,
                "high": high_count
            }

            print(f"{video_name}: low={low_count}, none={none_count}, high={high_count}")
            processed_files += 1

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {filename}: {e}")
            skipped_files.append(filename)
        except KeyError as e:
            print(f"Error: Missing key {e} in {filename}")
            skipped_files.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_files.append(filename)

    print("-" * 60)
    print(f"Processing complete:")
    print(f"  Successfully processed: {processed_files} files")
    if skipped_files:
        print(f"  Skipped files: {len(skipped_files)}")
        for skipped in skipped_files:
            print(f"    - {skipped}")

    # Save aggregated results
    try:
        with open(output_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        print(f"\nAggregated results saved to: {output_file}")

        # Print summary statistics
        total_low = sum(result['low'] for result in aggregated_results.values())
        total_none = sum(result['none'] for result in aggregated_results.values())
        total_high = sum(result['high'] for result in aggregated_results.values())
        total_detections = total_low + total_none + total_high

        print(f"\nSummary Statistics:")
        print(f"  Total videos: {len(aggregated_results)}")
        print(f"  Total detections: {total_detections}")
        print(f"    Low cards: {total_low}")
        print(f"    None cards: {total_none}")
        print(f"    High cards: {total_high}")

    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")
        sys.exit(1)

    return aggregated_results

def main():
    parser = argparse.ArgumentParser(description='Aggregate SegFormer results for evaluation')
    parser.add_argument('--input', '-i',
                       default='/home/cheskett/repos/cs231/cs231n_eye-in-the-sky/SEGFORMER_RESULTS/segformer_results',
                       help='Path to folder containing SegFormer JSON result files')
    parser.add_argument('--output', '-o',
                       default='segformer_aggregated_class_counts.json',
                       help='Output file path for aggregated results')

    args = parser.parse_args()

    # Aggregate the results
    results = aggregate_segformer_results(args.input, args.output)

    print(f"\nYou can now evaluate these results against ground truth using:")
    print(f"python utils/evaluate_predictions.py <ground_truth.json> {args.output}")

if __name__ == "__main__":
    main()