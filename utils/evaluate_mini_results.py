#!/usr/bin/env python3

import json
import argparse
import sys
from pathlib import Path

def load_json_file(file_path):
    """Load and validate JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        sys.exit(1)

def calculate_hilo_count(class_counts):
    """Calculate Hi-Lo count: +1 for low, -1 for high, 0 for none"""
    low_count = class_counts.get('low', 0)
    high_count = class_counts.get('high', 0)
    # none doesn't affect the count
    return low_count - high_count

def extract_class_counts(entry):
    """Extract class counts from entry (handles both predicted and ground_truth formats)"""
    if 'classes' in entry:
        return entry['classes']
    elif 'predicted' in entry and 'classes' in entry['predicted']:
        return entry['predicted']['classes']
    elif 'ground_truth' in entry:
        return entry['ground_truth']
    else:
        # Assume it's already in the right format
        return entry

def evaluate_mini_results(data):
    """Evaluate mini results comparing .mp4 (ground truth) with number-only (predictions)"""

    video_analysis = data.get('video_analysis', {})

    # Initialize counters
    total_false_positives = {'low': 0, 'high': 0, 'none': 0}
    total_false_negatives = {'low': 0, 'high': 0, 'none': 0}
    total_true_positives = {'low': 0, 'high': 0, 'none': 0}

    recalculated_video_results = {}
    recalculated_hilo_comparisons = {}

    # Find all video numbers (those without .mp4 extension)
    video_numbers = set()
    for key in video_analysis.keys():
        if not key.endswith('.mp4'):
            video_numbers.add(key)

    print("="*80)
    print("DETAILED VIDEO-BY-VIDEO ANALYSIS")
    print("="*80)

    for video_num in sorted(video_numbers, key=lambda x: int(x) if x.isdigit() else float('inf')):
        gt_key = f"{video_num}.mp4"
        pred_key = video_num

        # Skip if either ground truth or prediction is missing
        if gt_key not in video_analysis or pred_key not in video_analysis:
            print(f"WARNING: Missing data for video {video_num}")
            continue

        # Get ground truth counts
        gt_entry = video_analysis[gt_key]
        gt_counts = extract_class_counts(gt_entry)

        # Get predicted counts
        pred_entry = video_analysis[pred_key]
        pred_counts = extract_class_counts(pred_entry)

        # Calculate metrics for this video
        video_fp = {}
        video_fn = {}
        video_tp = {}

        for class_type in ['low', 'high', 'none']:
            gt_count = gt_counts.get(class_type, 0)
            pred_count = pred_counts.get(class_type, 0)

            # True positives: minimum of ground truth and predicted
            tp = min(gt_count, pred_count)
            # False positives: predicted - true positives
            fp = pred_count - tp
            # False negatives: ground truth - true positives
            fn = gt_count - tp

            video_fp[class_type] = fp
            video_fn[class_type] = fn
            video_tp[class_type] = tp

            # Add to totals
            total_false_positives[class_type] += fp
            total_false_negatives[class_type] += fn
            total_true_positives[class_type] += tp

        recalculated_video_results[gt_key] = {
            'ground_truth': gt_counts,
            'predicted': pred_counts,
            'false_positives': video_fp,
            'false_negatives': video_fn,
            'true_positives': video_tp
        }

        # Calculate Hi-Lo counts
        gt_hilo = calculate_hilo_count(gt_counts)
        pred_hilo = calculate_hilo_count(pred_counts)
        hilo_error = pred_hilo - gt_hilo

        recalculated_hilo_comparisons[gt_key] = {
            'ground_truth_hilo': gt_hilo,
            'predicted_hilo': pred_hilo,
            'hilo_error': hilo_error
        }

        # Print video details
        print(f"\n{gt_key}:")
        print(f"  Ground Truth: low={gt_counts.get('low', 0)}, high={gt_counts.get('high', 0)}, none={gt_counts.get('none', 0)} | Hi-Lo: {gt_hilo:+d}")
        print(f"  Predicted:    low={pred_counts.get('low', 0)}, high={pred_counts.get('high', 0)}, none={pred_counts.get('none', 0)} | Hi-Lo: {pred_hilo:+d}")
        print(f"  Errors:       FP={sum(video_fp.values())}, FN={sum(video_fn.values())} | Hi-Lo Error: {hilo_error:+d}")

        if sum(video_fp.values()) > 0 or sum(video_fn.values()) > 0:
            print(f"    False Positives: low={video_fp['low']}, high={video_fp['high']}, none={video_fp['none']}")
            print(f"    False Negatives: low={video_fn['low']}, high={video_fn['high']}, none={video_fn['none']}")

    return recalculated_video_results, recalculated_hilo_comparisons, total_false_positives, total_false_negatives, total_true_positives

def print_summary(total_fp, total_fn, total_tp, hilo_comparisons):
    """Print summary statistics"""

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Overall accuracy metrics
    print("\nCLASS-WISE ACCURACY:")
    print("-" * 40)
    for class_type in ['low', 'high', 'none']:
        tp = total_tp[class_type]
        fp = total_fp[class_type]
        fn = total_fn[class_type]

        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{class_type.upper()}:")
        print(f"  True Positives:  {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print()

    # Overall totals
    total_tp_all = sum(total_tp.values())
    total_fp_all = sum(total_fp.values())
    total_fn_all = sum(total_fn.values())

    overall_precision = total_tp_all / (total_tp_all + total_fp_all) if (total_tp_all + total_fp_all) > 0 else 0
    overall_recall = total_tp_all / (total_tp_all + total_fn_all) if (total_tp_all + total_fn_all) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("OVERALL ACCURACY:")
    print("-" * 40)
    print(f"Total True Positives:  {total_tp_all}")
    print(f"Total False Positives: {total_fp_all}")
    print(f"Total False Negatives: {total_fn_all}")
    print(f"Overall Precision: {overall_precision:.3f}")
    print(f"Overall Recall:    {overall_recall:.3f}")
    print(f"Overall F1-Score:  {overall_f1:.3f}")

    # Hi-Lo count analysis
    print("\nHI-LO COUNT ANALYSIS:")
    print("-" * 40)

    hilo_errors = [comp['hilo_error'] for comp in hilo_comparisons.values()]
    perfect_hilo_count = sum(1 for error in hilo_errors if error == 0)
    total_videos = len(hilo_errors)

    mean_hilo_error = sum(abs(error) for error in hilo_errors) / len(hilo_errors) if hilo_errors else 0
    max_hilo_error = max(abs(error) for error in hilo_errors) if hilo_errors else 0

    print(f"Perfect Hi-Lo Predictions: {perfect_hilo_count}/{total_videos} ({perfect_hilo_count/total_videos*100:.1f}%)")
    print(f"Mean Absolute Hi-Lo Error: {mean_hilo_error:.2f}")
    print(f"Maximum Hi-Lo Error: {max_hilo_error}")

    # Show Hi-Lo error distribution
    error_counts = {}
    for error in hilo_errors:
        error_counts[error] = error_counts.get(error, 0) + 1

    print(f"\nHi-Lo Error Distribution:")
    for error in sorted(error_counts.keys()):
        count = error_counts[error]
        print(f"  Error {error:+d}: {count} videos")

def save_corrected_results(video_results, hilo_comparisons, output_file):
    """Save corrected/recalculated results to JSON file"""
    corrected_results = {
        'video_analysis': video_results,
        'hilo_analysis': hilo_comparisons,
        'metadata': {
            'total_videos': len(video_results),
            'perfect_hilo_predictions': sum(1 for comp in hilo_comparisons.values() if comp['hilo_error'] == 0)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(corrected_results, f, indent=2)

    print(f"\nCorrected results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate mini_results.json comparing .mp4 entries (ground truth) with number entries (predictions)')
    parser.add_argument('mini_results', help='Path to mini_results.json file')
    parser.add_argument('--output', '-o', help='Save corrected/recalculated results to JSON file')

    args = parser.parse_args()

    # Load data file
    print("Loading mini results file...")
    data = load_json_file(args.mini_results)

    video_analysis = data.get('video_analysis', {})

    # Count ground truth and prediction entries
    gt_count = sum(1 for key in video_analysis.keys() if key.endswith('.mp4'))
    pred_count = sum(1 for key in video_analysis.keys() if not key.endswith('.mp4'))

    print(f"Ground truth entries (.mp4): {gt_count}")
    print(f"Prediction entries (numbers): {pred_count}")

    # Evaluate predictions
    video_results, hilo_comparisons, total_fp, total_fn, total_tp = evaluate_mini_results(data)

    # Print summary
    print_summary(total_fp, total_fn, total_tp, hilo_comparisons)

    # Save corrected results if requested
    if args.output:
        save_corrected_results(video_results, hilo_comparisons, args.output)

if __name__ == "__main__":
    main()