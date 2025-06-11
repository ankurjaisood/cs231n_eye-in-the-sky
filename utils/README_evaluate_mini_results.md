# Evaluate Mini Results Script

This script (`evaluate_mini_results.py`) is designed to evaluate predictions stored in the `mini_results.json` format, where:

- **Ground truth data** is stored in entries named like `"1.mp4"`, `"2.mp4"`, `"3.mp4"`, etc.
- **Prediction data** is stored in entries named like `"1"`, `"2"`, `"3"`, etc. (without the `.mp4` extension)

## Usage

### Basic Usage
```bash
python utils/evaluate_mini_results.py deep_sort_pytorch/mini_results.json
```

### Save Corrected Results
```bash
python utils/evaluate_mini_results.py deep_sort_pytorch/mini_results.json --output corrected_results.json
```

## Input Format

The script expects a JSON file with the following structure:

```json
{
  "video_analysis": {
    "1": {
      "predicted": {
        "classes": {
          "low": 1,
          "none": 1,
          "high": 2
        }
      }
    },
    "1.mp4": {
      "ground_truth": {
        "low": 1,
        "none": 1,
        "high": 2
      }
    }
  }
}
```

## Output

The script provides:

1. **Detailed video-by-video analysis** showing:
   - Ground truth vs predicted counts for each class (low, high, none)
   - Hi-Lo count comparison
   - False positives, false negatives, and errors

2. **Summary statistics** including:
   - Class-wise precision, recall, and F1-score
   - Overall accuracy metrics
   - Hi-Lo count analysis with error distribution

3. **Optional corrected results file** (when using `--output`) that:
   - Recalculates all metrics based on proper ground truth vs prediction comparison
   - Saves results in the same format as other evaluation scripts

## Metrics Explained

### Class-wise Metrics
- **True Positives (TP)**: Correctly predicted instances of each class
- **False Positives (FP)**: Incorrectly predicted instances (over-prediction)
- **False Negatives (FN)**: Missed instances (under-prediction)
- **Precision**: TP / (TP + FP) - accuracy of positive predictions
- **Recall**: TP / (TP + FN) - coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall

### Hi-Lo Analysis
- **Hi-Lo Count**: `low_count - high_count` (none cards don't affect this)
- **Hi-Lo Error**: `predicted_hilo - ground_truth_hilo`
- **Perfect Hi-Lo Predictions**: Videos where Hi-Lo error = 0

## Example Output

```
================================================================================
DETAILED VIDEO-BY-VIDEO ANALYSIS
================================================================================

1.mp4:
  Ground Truth: low=1, high=2, none=1 | Hi-Lo: -1
  Predicted:    low=1, high=2, none=1 | Hi-Lo: -1
  Errors:       FP=0, FN=0 | Hi-Lo Error: +0

================================================================================
SUMMARY STATISTICS
================================================================================

CLASS-WISE ACCURACY:
----------------------------------------
LOW:
  True Positives:  44
  False Positives: 6
  False Negatives: 1
  Precision: 0.880
  Recall:    0.978
  F1-Score:  0.926

OVERALL ACCURACY:
----------------------------------------
Total True Positives:  91
Total False Positives: 10
Total False Negatives: 18
Overall Precision: 0.901
Overall Recall:    0.835
Overall F1-Score:  0.867

HI-LO COUNT ANALYSIS:
----------------------------------------
Perfect Hi-Lo Predictions: 7/20 (35.0%)
Mean Absolute Hi-Lo Error: 1.05
Maximum Hi-Lo Error: 3
```

## Differences from Original evaluate_predictions.py

- **Input format**: Works with combined ground truth + predictions in single file
- **Data extraction**: Automatically pairs `.mp4` entries (ground truth) with number entries (predictions)
- **Error handling**: Handles missing data gracefully with warnings
- **Output format**: Compatible with existing evaluation result formats when using `--output`