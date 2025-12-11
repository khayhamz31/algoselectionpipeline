#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_metatargets(metatarget_path: str, metric: str = "accuracy") -> pd.DataFrame:
    """Load metatarget file (accuracy.csv or f1.csv)."""
    if not os.path.exists(metatarget_path):
        raise FileNotFoundError(f"Metatarget file not found: {metatarget_path}")
    df = pd.read_csv(metatarget_path, index_col=0)
    if df.empty:
        raise ValueError(f"Metatarget file is empty: {metatarget_path}")
    return df


def create_targets(
    metatarget_df: pd.DataFrame,
    algorithms: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, float, LabelEncoder]:
    """Create target labels based on best algorithm per dataset."""
    if algorithms:
        missing_algos = set(algorithms) - set(metatarget_df.columns)
        if missing_algos:
            raise ValueError(f"Algorithms not found in metatarget: {missing_algos}")
        perf_df = metatarget_df[algorithms]
    else:
        perf_df = metatarget_df.copy()

    targets, valid_datasets = [], []
    for dataset_id, row in perf_df.iterrows():
        if row.max() == 0:
            continue
        best_algorithm = row.idxmax()
        targets.append(best_algorithm)
        valid_datasets.append(dataset_id)

    if not valid_datasets:
        raise ValueError("No valid datasets found (all algorithms scored 0)")

    target_df = pd.DataFrame({
        'dataset_id': valid_datasets,
        'best_algorithm': targets
    }).set_index('dataset_id')

    le = LabelEncoder()
    target_df['best_algorithm_encoded'] = le.fit_transform(target_df['best_algorithm'])

    majority_class = target_df["best_algorithm"].value_counts().idxmax()
    majority_count = target_df["best_algorithm"].value_counts().max()
    majority_accuracy = majority_count / len(target_df)

    return target_df, majority_accuracy, le


def load_metafeatures() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine D2V and traditional meta-features."""
    d2v = pd.read_csv("qualities/d2v/metafeatures.csv", index_col=0)
    traditional = pd.read_csv("qualities/traditional/metafeatures.csv", index_col=0)
    hybrid = pd.concat([traditional, d2v], axis=1, join='inner')
    return d2v, traditional, hybrid


def align_features_with_targets(
    metafeature_dict: Dict[str, pd.DataFrame],
    target_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Align meta-features with targets (inner join) and append target column."""
    aligned = {}
    target_column = target_df["best_algorithm_encoded"]

    for name, mf_df in metafeature_dict.items():
        aligned_df = mf_df.join(target_column, how="inner")
        if aligned_df.empty:
            raise ValueError(f"No overlap between {name} meta-features and targets")
        aligned[name] = aligned_df

    return aligned


def evaluate_dataset(df: pd.DataFrame, seed: int) -> Tuple[float, np.ndarray]:
    """Evaluate dataset with Leave-One-Out CV using RandomForest."""
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    loo = LeaveOneOut()
    clf = RandomForestClassifier(random_state=seed)

    correct = 0
    fi_sum = np.zeros(X.shape[1], dtype=float)
    fi_count = 0

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        correct += int(y_pred[0] == y_test[0])
        if hasattr(clf, "feature_importances_"):
            fi_sum += clf.feature_importances_
            fi_count += 1

    accuracy = correct / len(X)
    mean_fi = (fi_sum / fi_count) if fi_count > 0 else np.zeros(X.shape[1])
    return accuracy, mean_fi


def run_evaluation(
    datasets: Dict[str, pd.DataFrame],
    n_repeats: int = 10,
    seed: int = 175,
    save_dir: Optional[str] = None
) -> Dict[str, List[float]]:
    """Run evaluation on multiple datasets with multiple repetitions."""
    results = {}
    if save_dir:
        ensure_dir(save_dir)

    for name in sorted(datasets.keys()):
        if len(datasets[name]) < 2:
            raise ValueError(f"Dataset '{name}' has fewer than 2 samples")
        scores = []
        for rep in tqdm(range(n_repeats), desc=f"Evaluating {name}", leave=False):
            current_seed = seed + rep
            acc, _ = evaluate_dataset(datasets[name], current_seed)
            scores.append(acc)
        results[name] = scores

        if save_dir:
            df_results = pd.DataFrame({
                "repetition": list(range(1, n_repeats + 1)),
                "accuracy": scores
            })
            df_results.to_csv(os.path.join(save_dir, f"{name}_results.csv"), index=False)

    return results


def plot_results(
    results: Dict[str, List[float]],
    majority_accuracy: float,
    title: str = "Meta-Learning Algorithm Selection Accuracy (LOO-CV, 10 repeats)",
    figsize: Tuple[int, int] = (12, 7)
):
    """Clean zoomed-in boxplot ensuring the majority baseline is visible."""
    fig, ax = plt.subplots(figsize=figsize)

    box_data = [results["d2v"], results["traditional"], results["hybrid"]]
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
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xlabel("Meta-Feature Type", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim([y_lower, y_upper])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    return fig, ax




def summarize_results(results: Dict[str, List[float]]) -> pd.DataFrame:
    """Summarize mean and std of results per feature set."""
    summary = []
    for name in sorted(results.keys()):
        summary.append({
            'Feature Set': name.capitalize(),
            'Mean Accuracy': np.mean(results[name]),
            'Std Dev': np.std(results[name]),
            'Min': np.min(results[name]),
            'Max': np.max(results[name])
        })
    df_summary = pd.DataFrame(summary)
    df_summary["Mean ± Std"] = (
        df_summary["Mean Accuracy"].round(3).astype(str)
        + " ± " +
        df_summary["Std Dev"].round(3).astype(str)
    )
    return df_summary


def run_meta_classifier(
    metric_name: str = "accuracy",
    algorithms: Optional[List[str]] = None,
    n_repeats: int = 10,
    seed: int = 175,
    output_dir: Optional[str] = None,
    plot_title: str = "Meta-Learning Algorithm Selection Accuracy (LOO-CV, 10 repeats)"
) -> Tuple[Dict[str, List[float]], float, pd.DataFrame, Tuple]:
    """Run the complete meta-classifier pipeline."""
    if metric_name not in ["accuracy", "f1"]:
        raise ValueError(f"metric_name must be 'accuracy' or 'f1', got '{metric_name}'")

    metatarget_path = os.path.join("meta_targets", f"{metric_name}.csv")

    if output_dir is None:
        algo_suffix = "_".join(sorted(algorithms)) if algorithms else "all_algorithms"
        output_dir = os.path.join("meta_classifier_results", f"{metric_name}_{algo_suffix}")

    ensure_dir(output_dir)

    metatarget_df = load_metatargets(metatarget_path, metric_name)
    target_df, majority_accuracy, label_encoder = create_targets(metatarget_df, algorithms)
    d2v, traditional, hybrid = load_metafeatures()

    metafeature_dict = {'d2v': d2v, 'traditional': traditional, 'hybrid': hybrid}
    aligned_datasets = align_features_with_targets(metafeature_dict, target_df)

    results = run_evaluation(aligned_datasets, n_repeats=n_repeats, seed=seed, save_dir=output_dir)
    summary_df = summarize_results(results)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Plot results
    fig, ax = plot_results(results, majority_accuracy, title=plot_title)

    # Display summary neatly
    print("\n📊 Summary of Meta-Learning Performance:")
    print(summary_df[["Feature Set", "Mean ± Std"]].to_string(index=False))

    return results, majority_accuracy, summary_df, (fig, ax)
