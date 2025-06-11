#!/usr/bin/env python3
"""
Process YOLOv5 validation results and create comprehensive metrics analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def parse_results_file(file_path):
    """
    Parse the all_classes.txt file and extract metrics data.

    Args:
        file_path (str): Path to the results file

    Returns:
        tuple: (overall_metrics, class_metrics_df)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip the header line and process data
    data_lines = []
    overall_metrics = None

    for line in lines:
        line = line.strip()
        if not line or 'Class' in line and 'Images' in line:  # Skip header
            continue

        # Split the line and extract relevant data
        parts = line.split()
        if len(parts) >= 7:  # Ensure we have enough columns
            try:
                class_name = parts[0]
                images = int(parts[1])
                instances = int(parts[2])
                precision = float(parts[3])
                recall = float(parts[4])
                map50 = float(parts[5])
                map50_95 = float(parts[6])

                row_data = {
                    'Class': class_name,
                    'Images': images,
                    'Instances': instances,
                    'Precision': precision,
                    'Recall': recall,
                    'mAP50': map50,
                    'mAP50-95': map50_95
                }

                if class_name == 'all':
                    overall_metrics = row_data
                else:
                    data_lines.append(row_data)

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: {line}")
                continue

    # Create DataFrame for individual classes
    class_metrics_df = pd.DataFrame(data_lines)

    return overall_metrics, class_metrics_df


def create_metrics_summary(overall_metrics, class_metrics_df):
    """
    Create a comprehensive metrics summary table.

    Args:
        overall_metrics (dict): Overall performance metrics
        class_metrics_df (DataFrame): Individual class metrics

    Returns:
        DataFrame: Summary statistics table
    """
    metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']

    summary_data = []

    for metric in metrics:
        metric_values = class_metrics_df[metric].values

        # Find min and max classes
        min_idx = np.argmin(metric_values)
        max_idx = np.argmax(metric_values)

        min_class = class_metrics_df.iloc[min_idx]['Class']
        max_class = class_metrics_df.iloc[max_idx]['Class']

        summary_row = {
            'Metric': metric,
            'Average': overall_metrics[metric],
            'Min_Value': metric_values[min_idx],
            'Min_Class': min_class,
            'Max_Value': metric_values[max_idx],
            'Max_Class': max_class,
            'Std_Dev': np.std(metric_values),
            'Range': metric_values[max_idx] - metric_values[min_idx]
        }

        summary_data.append(summary_row)

    return pd.DataFrame(summary_data)


def create_detailed_analysis(class_metrics_df):
    """
    Create detailed per-class analysis with rankings.

    Args:
        class_metrics_df (DataFrame): Individual class metrics

    Returns:
        DataFrame: Detailed analysis with rankings
    """
    metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']

    # Create a copy for analysis
    analysis_df = class_metrics_df.copy()

    # Add rankings for each metric
    for metric in metrics:
        analysis_df[f'{metric}_Rank'] = analysis_df[metric].rank(ascending=False, method='min').astype(int)

    # Add overall performance score (weighted average)
    weights = {'Precision': 0.2, 'Recall': 0.2, 'mAP50': 0.3, 'mAP50-95': 0.3}
    analysis_df['Overall_Score'] = sum(analysis_df[metric] * weight for metric, weight in weights.items())
    analysis_df['Overall_Rank'] = analysis_df['Overall_Score'].rank(ascending=False, method='min').astype(int)

    # Sort by overall performance
    analysis_df = analysis_df.sort_values('Overall_Score', ascending=False)

    return analysis_df


def print_summary_table(summary_df):
    """Print a nicely formatted summary table."""
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)

    # Format the table
    pd.set_option('display.float_format', '{:.3f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print(summary_df.to_string(index=False))

    print("\n" + "="*80)


def print_top_bottom_classes(analysis_df, n=5):
    """Print top and bottom performing classes."""
    metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']

    print(f"\nTOP {n} AND BOTTOM {n} CLASSES BY METRIC")
    print("="*60)

    for metric in metrics:
        print(f"\n{metric}:")
        print("-" * 30)

        # Top classes
        top_classes = analysis_df.nsmallest(n, f'{metric}_Rank')[['Class', metric, f'{metric}_Rank']]
        print(f"Top {n}:")
        for _, row in top_classes.iterrows():
            print(f"  {row['Class']}: {row[metric]:.3f} (Rank {row[f'{metric}_Rank']})")

        # Bottom classes
        bottom_classes = analysis_df.nlargest(n, f'{metric}_Rank')[['Class', metric, f'{metric}_Rank']]
        print(f"Bottom {n}:")
        for _, row in bottom_classes.iterrows():
            print(f"  {row['Class']}: {row[metric]:.3f} (Rank {row[f'{metric}_Rank']})")


def save_detailed_results(summary_df, analysis_df, output_dir="./"):
    """Save detailed results to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save summary
    summary_path = output_path / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMetrics summary saved to: {summary_path}")

    # Save detailed analysis
    analysis_path = output_path / "detailed_class_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"Detailed analysis saved to: {analysis_path}")

    # Create a performance report
    report_path = output_path / "performance_report.txt"
    with open(report_path, 'w') as f:
        f.write("YOLOv5 Performance Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("TOP 10 OVERALL PERFORMERS\n")
        f.write("-" * 30 + "\n")
        top_performers = analysis_df.head(10)[['Class', 'Overall_Score', 'Overall_Rank', 'Precision', 'Recall', 'mAP50', 'mAP50-95']]
        f.write(top_performers.to_string(index=False))
        f.write("\n\n")

        f.write("BOTTOM 10 OVERALL PERFORMERS\n")
        f.write("-" * 30 + "\n")
        bottom_performers = analysis_df.tail(10)[['Class', 'Overall_Score', 'Overall_Rank', 'Precision', 'Recall', 'mAP50', 'mAP50-95']]
        f.write(bottom_performers.to_string(index=False))

    print(f"Performance report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Process YOLOv5 validation results')
    parser.add_argument('--input', '-i', type=str, default='yolov5/all_classes.txt',
                       help='Path to the all_classes.txt file')
    parser.add_argument('--output', '-o', type=str, default='./',
                       help='Output directory for results')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of top/bottom classes to show')

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return

    print(f"Processing results from: {args.input}")

    # Parse the results file
    overall_metrics, class_metrics_df = parse_results_file(args.input)

    if overall_metrics is None or class_metrics_df.empty:
        print("Error: Could not parse results file or no data found!")
        return

    print(f"Found {len(class_metrics_df)} classes in results")

    # Create summary analysis
    summary_df = create_metrics_summary(overall_metrics, class_metrics_df)

    # Create detailed analysis
    analysis_df = create_detailed_analysis(class_metrics_df)

    # Print results
    print_summary_table(summary_df)
    print_top_bottom_classes(analysis_df, args.top_n)

    # Save results
    save_detailed_results(summary_df, analysis_df, args.output)

    print(f"\nProcessing complete! Check {args.output} for detailed results.")


if __name__ == "__main__":
    main()