#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def plot_regressor_as_classifier(
    accuracies: Dict[str, List[float]],
    majority_accuracy: float,
    title: str = "Meta-Regressor as Classifier: Top-1 Accuracy (LOO-CV, 10 repeats)",
    ylabel: str = "Mean Accuracy",
    figsize: Tuple[int, int] = (12, 7)
) -> Tuple:
    """Create boxplot comparing regressor top-1 accuracy across feature sets."""
    fig, ax = plt.subplots(figsize=figsize)

    box_data = [accuracies["d2v"], accuracies["traditional"], accuracies["hybrid"]]
    labels = ["Dataset2Vec", "Traditional", "Hybrid"]

    # Simple black-and-white boxplot
    bp = ax.boxplot(
        box_data,
        labels=labels,
        patch_artist=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(linewidth=1.5, color="black"),
        whiskerprops=dict(linewidth=1.5, color="black"),
        capprops=dict(linewidth=1.5, color="black"),
        flierprops=dict(marker='o', markersize=5, color="black", alpha=0.6)
    )

    # Compute zoomed range ensuring baseline is included
    all_values = np.concatenate(box_data)
    data_min, data_max = np.min(all_values), np.max(all_values)
    lower_bound = min(data_min, majority_accuracy)
    upper_bound = max(data_max, majority_accuracy)
    margin = (upper_bound - lower_bound) * 0.25
    y_lower = max(0, lower_bound - margin)
    y_upper = min(1, upper_bound + margin)

    # Baseline
    ax.axhline(
        y=majority_accuracy,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Majority Baseline ({majority_accuracy:.2%})"
    )

    # Titles and styling
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Meta-Feature Type", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim([y_lower, y_upper])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    return fig, ax


def summarize_accuracies(accuracies: Dict[str, List[float]]) -> pd.DataFrame:
    """Summarize mean and std of top-1 accuracies per feature set."""
    summary = []
    for name in sorted(accuracies.keys()):
        summary.append({
            'Feature Set': name.capitalize(),
            'Mean Accuracy': np.mean(accuracies[name]),
            'Std Dev': np.std(accuracies[name]),
            'Min': np.min(accuracies[name]),
            'Max': np.max(accuracies[name])
        })
    df_summary = pd.DataFrame(summary)
    df_summary["Mean ± Std"] = (
        df_summary["Mean Accuracy"].round(3).astype(str)
        + " ± " +
        df_summary["Std Dev"].round(3).astype(str)
    )
    return df_summary


def plot_regressor_as_classifier_results(
    analysis_csv: str = "meta_regressor_results/analysis/top1_accuracy_analysis.csv",
    output_dir: Optional[str] = None,
    plot_title: str = "Meta-Regressor as Classifier: Top-1 Accuracy (LOO-CV, 10 repeats)",
    ylabel: str = "Mean Accuracy"
) -> Tuple[Dict[str, List[float]], float, pd.DataFrame, Tuple]:
    """
    Load top-1 accuracy from analysis CSV and plot results.
    
    Args:
        analysis_csv: Path to top1_accuracy_analysis.csv
        output_dir: Directory to save results (defaults to parent of analysis_csv)
        plot_title: Title for the plot
        ylabel: Y-axis label (default: "Mean Accuracy")
    
    Returns:
        Tuple of (accuracies_dict, majority_accuracy, summary_df, (fig, ax))
    """
    if not os.path.exists(analysis_csv):
        raise FileNotFoundError(f"Analysis CSV not found: {analysis_csv}")
    
    # Load the analysis results
    df = pd.read_csv(analysis_csv)
    
    if output_dir is None:
        output_dir = os.path.dirname(analysis_csv)
    
    # Extract accuracies per feature set
    accuracies = {
        'd2v': df[df['featureset'] == 'd2v']['top_1_accuracy'].tolist(),
        'traditional': df[df['featureset'] == 'traditional']['top_1_accuracy'].tolist(),
        'hybrid': df[df['featureset'] == 'hybrid']['top_1_accuracy'].tolist()
    }
    
    # Get majority baseline from the CSV (should be same for all rows)
    majority_accuracy = df['majority_baseline'].iloc[0]
    print(f"Majority Baseline Accuracy: {majority_accuracy:.4f}")
    
    # Summarize results
    summary_df = summarize_accuracies(accuracies)
    
    # Plot results
    print("\nGenerating boxplot...")
    fig, ax = plot_regressor_as_classifier(accuracies, majority_accuracy, title=plot_title, ylabel=ylabel)
    
    # Display summary with Mean ± Std format
    print("\n📊 Summary of Meta-Regressor Top-1 Accuracy (as Classifier):")
    for idx, row in summary_df.iterrows():
        print(f"{row['Feature Set']}: {row['Mean Accuracy']:.4f} ± {row['Std Dev']:.4f}")
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "regressor_as_classifier_boxplot.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    return accuracies, majority_accuracy, summary_df, (fig, ax)


if __name__ == "__main__":
    accuracies, majority_accuracy, summary_df, (fig, ax) = plot_regressor_as_classifier_results()
    plt.show()