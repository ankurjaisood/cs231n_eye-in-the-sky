#!/usr/bin/env python3
"""
Create histogram from Hi-Lo Error Distribution data in JSON evaluation results
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import sys
import os

def calculate_hilo_count(class_counts):
    """Calculate Hi-Lo count: +1 for low, -1 for high, 0 for none"""
    low_count = class_counts.get('low', 0)
    high_count = class_counts.get('high', 0)
    return low_count - high_count

def extract_hilo_errors_from_json(json_file):
    """Extract Hi-Lo errors from evaluation results JSON"""

    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if 'hilo_analysis' in data:
            # Use the pre-calculated hilo analysis
            hilo_errors = [video_data['hilo_error'] for video_data in data['hilo_analysis'].values()]
        elif 'video_analysis' in data:
            # Calculate from video analysis data
            hilo_errors = []
            for video_data in data['video_analysis'].values():
                gt_hilo = calculate_hilo_count(video_data['ground_truth'])
                pred_hilo = calculate_hilo_count(video_data['predicted'])
                hilo_error = pred_hilo - gt_hilo
                hilo_errors.append(hilo_error)
        else:
            print(f"Error: Could not find hilo_analysis or video_analysis in {json_file}")
            sys.exit(1)

        return hilo_errors

    except Exception as e:
        print(f"Error reading file {json_file}: {e}")
        sys.exit(1)

def create_hilo_error_histogram(hilo_errors, output_prefix="hilo_error_histogram"):
    """Create histogram from Hi-Lo error list"""

    # Count occurrences of each error value
    error_counts = {}
    for error in hilo_errors:
        error_counts[error] = error_counts.get(error, 0) + 1

    # Sort by error value for proper display
    sorted_errors = sorted(error_counts.keys())
    video_counts = [error_counts[error] for error in sorted_errors]

    # Create the histogram
    plt.figure(figsize=(10, 6))

    # Create bar plot (histogram-like)
    bars = plt.bar(sorted_errors, video_counts, width=0.8, alpha=0.7, color='steelblue', edgecolor='black')

    # Customize the plot
    plt.xlabel('Hi-Lo Count Error', fontsize=12)
    plt.ylabel('Number of Videos', fontsize=12)
    plt.title('Hi-Lo Error Distribution Histogram', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on top of bars
    for bar, count in zip(bars, video_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')

    # Set x-axis ticks to show all error values
    plt.xticks(sorted_errors)

    # Set y-axis to start from 0 and have nice spacing
    plt.ylim(0, max(video_counts) + 1)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the histogram
    png_file = f"{output_prefix}.png"
    pdf_file = f"{output_prefix}.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')

    # Show the plot
    plt.show()

    print("Histogram saved as:")
    print(f"- {png_file}")
    print(f"- {pdf_file}")

    # Print some additional statistics
    total_videos = len(hilo_errors)
    perfect_predictions = hilo_errors.count(0)
    accuracy_percentage = (perfect_predictions / total_videos) * 100 if total_videos > 0 else 0

    print(f"\nStatistics:")
    print(f"Total videos analyzed: {total_videos}")
    print(f"Perfect predictions (error = 0): {perfect_predictions} ({accuracy_percentage:.1f}%)")

    # Calculate mean absolute error and mean error
    mean_abs_error = sum(abs(error) for error in hilo_errors) / total_videos if total_videos > 0 else 0
    mean_error = sum(hilo_errors) / total_videos if total_videos > 0 else 0

    print(f"Mean absolute error: {mean_abs_error:.2f}")
    print(f"Mean error: {mean_error:.2f}")

    # Print error distribution
    print(f"\nHi-Lo Error Distribution:")
    for error in sorted_errors:
        count = error_counts[error]
        print(f"  Error {error:+d}: {count} videos")

def main():
    parser = argparse.ArgumentParser(description='Create histogram from Hi-Lo Error data in JSON evaluation results')
    parser.add_argument('input_file', help='Path to the JSON evaluation results file')
    parser.add_argument('--output', '-o', default='hilo_error_histogram',
                       help='Output file prefix (default: hilo_error_histogram)')

    args = parser.parse_args()

    # Extract the Hi-Lo error data
    hilo_errors = extract_hilo_errors_from_json(args.input_file)

    print(f"Extracted {len(hilo_errors)} Hi-Lo error values from {args.input_file}")

    # Create the histogram
    create_hilo_error_histogram(hilo_errors, args.output)

if __name__ == "__main__":
    main()