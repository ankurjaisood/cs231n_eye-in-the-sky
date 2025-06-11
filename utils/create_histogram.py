#!/usr/bin/env python3
"""
Create histogram from Hi-Lo Error Distribution data
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import sys
import os

def parse_hilo_error_distribution(file_path):
    """Parse Hi-Lo Error Distribution from report file"""

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    errors = []
    video_counts = []

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find the Hi-Lo Error Distribution section
        if "Hi-Lo Error Distribution:" not in content:
            print(f"Error: 'Hi-Lo Error Distribution:' section not found in {file_path}")
            sys.exit(1)

        # Split content and find the error distribution section
        lines = content.split('\n')
        in_distribution_section = False

        for line in lines:
            if "Hi-Lo Error Distribution:" in line:
                in_distribution_section = True
                continue

            if in_distribution_section:
                # Look for lines like "  Error +0: 8 videos" or "  Error -1: 5 videos"
                match = re.match(r'\s*Error\s*([+-]?\d+):\s*(\d+)\s*videos?', line)
                if match:
                    error_value = int(match.group(1))
                    count = int(match.group(2))
                    errors.append(error_value)
                    video_counts.append(count)
                elif line.strip() == "" or line.startswith("="):
                    # End of section if we hit empty line or section separator
                    break

        if not errors:
            print(f"Error: No error distribution data found in {file_path}")
            sys.exit(1)

        # Sort by error value for proper display
        sorted_data = sorted(zip(errors, video_counts))
        errors, video_counts = zip(*sorted_data)

        return list(errors), list(video_counts)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def create_hilo_error_histogram(errors, video_counts, output_prefix="hilo_error_histogram"):
    """Create histogram from Hi-Lo error distribution data"""

    # Create the histogram
    plt.figure(figsize=(10, 6))

    # Create bar plot (histogram-like)
    bars = plt.bar(errors, video_counts, width=0.8, alpha=0.7, color='steelblue', edgecolor='black')

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
    plt.xticks(errors)

    # Set y-axis to start from 0 and have nice spacing
    plt.ylim(0, max(video_counts) + 1)

    # Add some statistics as text
    total_videos = sum(video_counts)
    perfect_predictions = video_counts[errors.index(0)] if 0 in errors else 0
    accuracy_percentage = (perfect_predictions / total_videos) * 100 if total_videos > 0 else 0

    # stats_text = f'Total Videos: {total_videos}\nPerfect Predictions (Error = 0): {perfect_predictions} ({accuracy_percentage:.1f}%)'
    # plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
    #          horizontalalignment='right', verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
    print(f"\nStatistics:")
    print(f"Total videos analyzed: {total_videos}")
    print(f"Perfect predictions (error = 0): {perfect_predictions} ({accuracy_percentage:.1f}%)")

    # Calculate mean absolute error and mean error
    mean_abs_error = sum(abs(error) * count for error, count in zip(errors, video_counts)) / total_videos if total_videos > 0 else 0
    mean_error = sum(error * count for error, count in zip(errors, video_counts)) / total_videos if total_videos > 0 else 0

    print(f"Mean absolute error: {mean_abs_error:.2f}")
    print(f"Mean error: {mean_error:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Create histogram from Hi-Lo Error Distribution in report file')
    parser.add_argument('input_file', help='Path to the report file containing Hi-Lo Error Distribution')
    parser.add_argument('--output', '-o', default='hilo_error_histogram',
                       help='Output file prefix (default: hilo_error_histogram)')

    args = parser.parse_args()

    # Parse the error distribution data
    errors, video_counts = parse_hilo_error_distribution(args.input_file)

    print(f"Parsed Hi-Lo Error Distribution from {args.input_file}:")
    for error, count in zip(errors, video_counts):
        print(f"  Error {error:+d}: {count} videos")
    print()

    # Create the histogram
    create_hilo_error_histogram(errors, video_counts, args.output)

if __name__ == "__main__":
    main()