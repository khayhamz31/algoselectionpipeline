#!/usr/bin/env python
# coding: utf-8

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from tqdm import tqdm
from typing import Tuple, Dict, Optional, List

# Suppress scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# LOCAL MODELS MAPPING
# ============================================================================

LOCAL_MODELS = {
    'random_forest': RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1, max_depth=10),
    'support_vector_machine': LinearSVC(random_state=42, max_iter=100, tol=1e-2, dual=False),
    'linear_models': LogisticRegression(max_iter=100, random_state=42, n_jobs=-1, solver='liblinear', tol=1e-3),
    'xgboost': GradientBoostingClassifier(n_estimators=10, random_state=42, max_depth=3, learning_rate=0.1),
    'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=8)
}


# ============================================================================
# LOGGING & UTILITIES
# ============================================================================

def log_info(message: str):
    """Print info message with [INFO] prefix."""
    print(f"[INFO] {message}")


def log_error(message: str):
    """Print error message with [ERROR] prefix."""
    print(f"[ERROR] {message}")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ============================================================================
# MISSING ACCURACY FILLING
# ============================================================================

def load_dataset_and_labels(dataset_id: str, data_root: str = "test_datasets") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset and labels from .dat files (saved with numpy.savetxt).
    
    Args:
        dataset_id: Name of the dataset directory
        data_root: Root directory containing datasets
    
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    dataset_path = os.path.join(data_root, dataset_id)
    data_file = os.path.join(dataset_path, f"{dataset_id}_py.dat")
    labels_file = os.path.join(dataset_path, "labels_py.dat")
    
    log_info(f"Loading dataset {dataset_id}...")
    log_info(f"  Data file: {data_file}")
    log_info(f"  Labels file: {labels_file}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    try:
        data = np.loadtxt(data_file, delimiter=',')
        log_info(f"  ✓ Data loaded: shape {data.shape}, dtype {data.dtype}")
        
        # Labels can be 1D (no delimiter) or 2D (comma-delimited)
        # Try loading with delimiter first (for 2D case)
        try:
            labels = np.loadtxt(labels_file, delimiter=',')
            log_info(f"  ✓ Labels loaded (with delimiter): shape {labels.shape}, dtype {labels.dtype}")
            # If shape is (n,) after loading with delimiter, it's actually 1D
            if labels.ndim == 1:
                labels = labels.astype(int)
                log_info(f"  → Converted to int: shape {labels.shape}")
        except (ValueError, TypeError) as e:
            log_info(f"  ⚠ Delimiter loading failed: {str(e)}")
            log_info(f"  → Attempting to load without delimiter...")
            # Fall back to loading without delimiter (1D case)
            labels = np.loadtxt(labels_file)
            log_info(f"  ✓ Labels loaded (no delimiter): shape {labels.shape}, dtype {labels.dtype}")
            labels = labels.astype(int)
            log_info(f"  → Converted to int: shape {labels.shape}")
        
        return data, labels
    except Exception as e:
        log_error(f"Failed to load dataset {dataset_id}: {str(e)}")
        raise


def compute_local_accuracy_for_missing_algos(
    dataset_id: str,
    missing_algos: List[str],
    data_root: str = "test_datasets",
    max_samples_per_fold: int = 2000
) -> Dict[str, float]:
    """
    Compute accuracy for missing algorithms using 5-fold stratified cross-validation.
    
    Args:
        dataset_id: Name of the dataset
        missing_algos: List of algorithm names to compute accuracy for
        data_root: Root directory containing datasets
        max_samples_per_fold: Max samples per fold (subsample if dataset too large, default 2000)
    
    Returns:
        Dictionary mapping algorithm names to mean accuracy scores
    """
    log_info(f"Starting accuracy computation for dataset {dataset_id}")
    log_info(f"  Missing algorithms: {missing_algos}")
    
    try:
        X, y = load_dataset_and_labels(dataset_id, data_root)
    except FileNotFoundError as e:
        log_error(f"Cannot compute accuracy for {dataset_id}: {str(e)}")
        return {algo: np.nan for algo in missing_algos}
    
    log_info(f"Dataset loaded: X.shape={X.shape}, y.shape={y.shape}")
    log_info(f"  X dtype: {X.dtype}, y dtype: {y.dtype}")
    log_info(f"  Unique labels: {np.unique(y)}")
    
    # Subsample if dataset is very large (to keep SVC tractable)
    if len(X) > max_samples_per_fold * 10:
        log_info(f"  ⚠ Dataset too large ({len(X)} samples), subsampling to {max_samples_per_fold * 10}...")
        X, _, y, _ = train_test_split(X, y, train_size=max_samples_per_fold * 10, 
                                       stratify=y, random_state=42)
        log_info(f"  → Subsampled to: X.shape={X.shape}")
    
    results = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    log_info(f"Using 3-fold Stratified K-Fold CV")
    
    for algo in missing_algos:
        log_info(f"\nTraining {algo}...")
        
        if algo not in LOCAL_MODELS:
            log_error(f"Unknown algorithm: {algo}")
            results[algo] = np.nan
            continue
        
        try:
            model = LOCAL_MODELS[algo]
            log_info(f"  Model: {model.__class__.__name__}")
            fold_scores = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                log_info(f"  Fold {fold_idx + 1}: train={len(train_idx)}, test={len(test_idx)}")
                
                # Clone model for this fold
                from sklearn.base import clone
                clf = clone(model)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                fold_acc = accuracy_score(y_test, y_pred)
                fold_scores.append(fold_acc)
                log_info(f"    → Fold accuracy: {fold_acc:.4f}")
            
            mean_accuracy = np.mean(fold_scores)
            std_accuracy = np.std(fold_scores)
            results[algo] = mean_accuracy
            log_info(f"  ✓ {algo}: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            
        except Exception as e:
            log_error(f"Failed to compute accuracy for {algo} on {dataset_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[algo] = np.nan
    
    log_info(f"Completed accuracy computation for dataset {dataset_id}\n")
    return results


def fill_missing_accuracies(
    meta_path: str = "meta_targets/accuracy.csv",
    data_root: str = "test_datasets"
) -> pd.DataFrame:
    """
    Detect and fill missing accuracy values in the metatarget CSV.
    Updates the file in place and returns the filled dataframe.
    
    Args:
        meta_path: Path to the accuracy CSV file
        data_root: Root directory containing datasets
    
    Returns:
        Updated dataframe with filled accuracies
    """
    # Load existing accuracy matrix
    if not os.path.exists(meta_path):
        log_error(f"Metatarget file not found: {meta_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(meta_path, index_col=0)
    except Exception as e:
        log_error(f"Failed to load {meta_path}: {str(e)}")
        return pd.DataFrame()
    
    if df.empty:
        log_info("Accuracy matrix is empty, skipping filling.")
        return df
    
    log_info(f"Loaded accuracy matrix: {df.shape}")
    log_info(f"  Columns: {list(df.columns)}")
    log_info(f"  Index: {list(df.index[:5])}{'...' if len(df.index) > 5 else ''}")
    log_info(f"Initial NaN count: {df.isna().sum().sum()}")
    log_info(f"  NaN per column: {dict(df.isna().sum())}")
    
    # Ensure all algorithm columns exist (add as NaN if missing)
    for algo in LOCAL_MODELS.keys():
        if algo not in df.columns:
            log_info(f"Adding missing algorithm column: {algo}")
            df[algo] = np.nan
    
    # Find cells with NaN
    missing_per_dataset = {}
    for dataset_id in df.index:
        dataset_id_str = str(dataset_id)  # Convert to string in case index is numeric
        missing_algos = [col for col in df.columns if pd.isna(df.loc[dataset_id, col])]
        if missing_algos:
            missing_per_dataset[dataset_id_str] = missing_algos
    
    if not missing_per_dataset:
        log_info("No missing accuracies found.")
        return df
    
    log_info(f"Found {len(missing_per_dataset)} datasets with missing accuracies:")
    for dataset_id, algos in list(missing_per_dataset.items())[:5]:
        log_info(f"  Dataset {dataset_id}: {algos}")
    if len(missing_per_dataset) > 5:
        log_info(f"  ... and {len(missing_per_dataset) - 5} more")
    
    # Fill missing accuracies
    filled_count = 0
    failed_count = 0
    for dataset_id_str, missing_algos in tqdm(missing_per_dataset.items(), desc="Filling missing accuracies"):
        log_info(f"\n{'='*70}")
        log_info(f"Dataset {dataset_id_str}: Computing {len(missing_algos)} algorithms")
        log_info(f"{'='*70}")
        
        accuracies = compute_local_accuracy_for_missing_algos(dataset_id_str, missing_algos, data_root)
        
        for algo, accuracy in accuracies.items():
            if not np.isnan(accuracy):
                # Try to find the matching index (could be str or int)
                for idx in df.index:
                    if str(idx) == dataset_id_str:
                        df.loc[idx, algo] = accuracy
                        filled_count += 1
                        log_info(f"✓ Updated {algo}: {accuracy:.4f}")
                        break
            else:
                failed_count += 1
                log_error(f"✗ Failed to compute {algo}")
    
    log_info(f"\n{'='*70}")
    log_info(f"Filled {filled_count} missing values ({failed_count} failures)")
    log_info(f"{'='*70}\n")
    
    # Save updated dataframe
    try:
        df.to_csv(meta_path)
        log_info(f"Updated accuracy matrix saved to {meta_path}")
        log_info(f"  Final shape: {df.shape}")
        log_info(f"  Remaining NaN count: {df.isna().sum().sum()}")
    except Exception as e:
        log_error(f"Failed to save accuracy matrix: {str(e)}")
    
    return df


# ============================================================================
# ORIGINAL META-REGRESSION FUNCTIONS
# ============================================================================

def load_metatargets(metatarget_path: str, metric: str = "accuracy") -> pd.DataFrame:
    """Load metatarget file (accuracy.csv or f1.csv)."""
    if not os.path.exists(metatarget_path):
        raise FileNotFoundError(f"Metatarget file not found: {metatarget_path}")
    df = pd.read_csv(metatarget_path, index_col=0)
    if df.empty:
        raise ValueError(f"Metatarget file is empty: {metatarget_path}")
    
    log_info(f"Loaded performance matrix: {df.shape}")
    log_info(f"Algorithms: {list(df.columns)}")
    
    # Filter out datasets where all algorithms score 0
    valid_mask = (df > 0).any(axis=1)
    df_filtered = df[valid_mask]
    excluded_count = len(df) - len(df_filtered)
    
    if excluded_count > 0:
        log_info(f"Excluded {excluded_count} datasets where all algorithms score 0")
    
    log_info(f"Valid datasets for regression: {len(df_filtered)}")
    log_info(f"Performance statistics:")
    print(df_filtered.describe())
    
    return df_filtered


def load_metafeatures() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine D2V and traditional meta-features."""
    d2v = pd.read_csv("qualities/d2v/metafeatures.csv", index_col=0)
    traditional = pd.read_csv("qualities/traditional/metafeatures.csv", index_col=0)
    hybrid = pd.concat([traditional, d2v], axis=1, join='inner')
    return d2v, traditional, hybrid


def align_features_with_targets(
    metafeature_dict: Dict[str, pd.DataFrame],
    target_df: pd.DataFrame
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Align meta-features with targets (inner join) for regression."""
    aligned = {}
    
    for name, mf_df in metafeature_dict.items():
        # Inner join to keep only datasets with both metafeatures and targets
        common_datasets = mf_df.index.intersection(target_df.index)
        
        X = mf_df.loc[common_datasets]
        y = target_df.loc[common_datasets]
        
        if len(X) == 0:
            raise ValueError(f"No overlap between {name} meta-features and targets")
        
        log_info(f"{name}: {len(X)} datasets, {X.shape[1]} features, {y.shape[1]} algorithms")
        aligned[name] = (X, y)
    
    return aligned


def compute_baseline(performance_matrix: pd.DataFrame) -> float:
    """Compute baseline MAE (mean predictor)."""
    mean_baseline = performance_matrix.mean()
    baseline_mae = mean_absolute_error(
        performance_matrix.values.flatten(),
        np.tile(mean_baseline.values, len(performance_matrix))
    )
    log_info(f"Baseline (mean predictor) MAE: {baseline_mae:.4f}")
    return baseline_mae


def evaluate_regression_dataset(X: pd.DataFrame, y: pd.DataFrame, seed: int) -> Dict:
    """
    Evaluate multi-output regression with LOO-CV using RandomForest.
    Returns: dict with metrics, feature importances, and predictions
    """
    if len(X) < 2:
        return {
            "mae": np.nan, "mse": np.nan, "r2": np.nan,
            "feature_importances": np.zeros(X.shape[1]),
            "predictions": None, "actual_values": None
        }
    
    X_values = X.values
    y_values = y.values
    feature_names = X.columns
    algorithm_names = y.columns

    loo = LeaveOneOut()
    regressor = RandomForestRegressor(random_state=seed)
    
    predictions = []
    actual_values = []
    fi_sum = np.zeros(len(feature_names), dtype=float)
    fi_count = 0

    for train_index, test_index in loo.split(X_values):
        X_train, X_test = X_values[train_index], X_values[test_index]
        y_train, y_test = y_values[train_index], y_values[test_index]

        regressor.fit(X_train, y_train)  # Multi-output regression
        y_pred = regressor.predict(X_test)  # Shape: [1, n_algorithms]
        
        predictions.append(y_pred[0])  # Single test sample
        actual_values.append(y_test[0])
        
        # Collect feature importances
        if hasattr(regressor, "feature_importances_"):
            fi_sum += regressor.feature_importances_
            fi_count += 1

    predictions = np.array(predictions)  # [n_datasets, n_algorithms]
    actual_values = np.array(actual_values)
    
    # Calculate regression metrics
    mae = mean_absolute_error(actual_values, predictions)
    mse = mean_squared_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    # Mean feature importances
    mean_fi = (fi_sum / fi_count) if fi_count > 0 else np.zeros(len(feature_names))
    
    return {
        "mae": mae,
        "mse": mse, 
        "r2": r2,
        "feature_importances": mean_fi,
        "predictions": predictions,
        "actual_values": actual_values,
        "algorithm_names": algorithm_names
    }


def run_regression_evaluation(
    datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    n_repeats: int = 10,
    seed: int = 175,
    save_dir: Optional[str] = None
) -> Dict:
    """Run regression evaluation on multiple metafeature types and save all predictions."""
    results = {}
    
    if save_dir:
        ensure_dir(save_dir)

    for name in sorted(datasets.keys()):
        log_info(f"Evaluating {name} meta-regression model...")
        X, y = datasets[name]
        
        mae_scores = []
        mse_scores = []
        r2_scores = []
        all_predictions = []
        all_actual_values = []
        algorithm_names = None
        per_algo_mae_list = []

        for rep in tqdm(range(n_repeats), desc=f"  {name}", leave=False):
            current_seed = seed + rep
            result = evaluate_regression_dataset(X, y, current_seed)
            
            mae_scores.append(result["mae"])
            mse_scores.append(result["mse"])
            r2_scores.append(result["r2"])
            
            # Collect predictions from each repetition
            if result["predictions"] is not None:
                all_predictions.append(result["predictions"])
                all_actual_values.append(result["actual_values"])
                if algorithm_names is None:
                    algorithm_names = result["algorithm_names"]
                
                # Calculate per-algorithm MAE for this repetition
                predictions = result["predictions"]
                actual_values = result["actual_values"]
                for i, algo in enumerate(algorithm_names):
                    algo_mae = mean_absolute_error(actual_values[:, i], predictions[:, i])
                    per_algo_mae_list.append({
                        'repetition': rep + 1,
                        'algorithm': algo,
                        'mae': algo_mae
                    })

        results[name] = {
            "mae": mae_scores,
            "mse": mse_scores,
            "r2": r2_scores
        }
        
        log_info(f"  MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        log_info(f"  MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        log_info(f"  R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

        if save_dir:
            # Save detailed results metrics
            results_df = pd.DataFrame({
                "repetition": list(range(1, n_repeats + 1)),
                "mae": mae_scores,
                "mse": mse_scores,
                "r2": r2_scores
            })
            results_df.to_csv(os.path.join(save_dir, f"{name}_results.csv"), index=False)
            
            # Save per-algorithm MAE
            if per_algo_mae_list:
                per_algo_df = pd.DataFrame(per_algo_mae_list)
                per_algo_df.to_csv(
                    os.path.join(save_dir, f"{name}_per_algorithm_mae.csv"), 
                    index=False
                )
                
                # Print per-algorithm MAE summary
                log_info(f"Per-algorithm MAE summary ({name}):")
                for algo in algorithm_names:
                    algo_data = per_algo_df[per_algo_df['algorithm'] == algo]['mae']
                    log_info(f"  {algo}: {algo_data.mean():.4f} ± {algo_data.std():.4f}")
            
            # Save all predictions from all repetitions
            if all_predictions:
                predictions_dir = os.path.join(save_dir, "predictions", name)
                ensure_dir(predictions_dir)
                
                for rep, (preds, actuals) in enumerate(zip(all_predictions, all_actual_values)):
                    # Create combined dataframe for this repetition
                    rep_df = pd.DataFrame()
                    for i, algo in enumerate(algorithm_names):
                        rep_df[f"{algo}_actual"] = actuals[:, i]
                        rep_df[f"{algo}_pred"] = preds[:, i]
                    
                    rep_df.to_csv(
                        os.path.join(predictions_dir, f"rep_{rep+1:02d}.csv"), 
                        index=False
                    )

    return results


def summarize_results(results: Dict) -> pd.DataFrame:
    """Summarize mean and std of results per feature set."""
    summary = []
    for name in sorted(results.keys()):
        summary.append({
            'Feature Set': name.capitalize(),
            'Mean MAE': np.mean(results[name]["mae"]),
            'Std MAE': np.std(results[name]["mae"]),
            'Mean MSE': np.mean(results[name]["mse"]),
            'Std MSE': np.std(results[name]["mse"]),
            'Mean R²': np.mean(results[name]["r2"]),
            'Std R²': np.std(results[name]["r2"])
        })
    return pd.DataFrame(summary)


def get_top_algorithm(row_values: np.ndarray, algorithm_names: list) -> str:
    """Get the algorithm with highest value in a row."""
    top_idx = np.argmax(row_values)
    return algorithm_names[top_idx]


def compute_majority_classifier_baseline(performance_matrix: pd.DataFrame) -> float:
    """
    Compute majority classifier baseline.
    Majority classifier always predicts the algorithm with best average performance.
    """
    # The "majority" algorithm is the one with highest average performance
    avg_performance = performance_matrix.mean()
    majority_algo = avg_performance.idxmax()
    
    log_info(f"Majority Classifier Baseline:")
    log_info(f"  Best overall algorithm: {majority_algo}")
    log_info(f"  Average performance: {avg_performance[majority_algo]:.4f}")
    
    # Calculate accuracy if we always predict the best algorithm
    actual_best_per_dataset = performance_matrix.idxmax(axis=1)
    accuracy = (actual_best_per_dataset == majority_algo).sum() / len(performance_matrix)
    
    log_info(f"  Majority classifier accuracy: {accuracy:.4f}")
    
    return accuracy


def analyze_top1_predictions(
    results_dir: str,
    performance_matrix: pd.DataFrame,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze top-1 accuracy across all feature sets and repetitions.
    Compares predicted top algorithm vs actual top algorithm.
    """
    
    if save_dir is None:
        save_dir = os.path.join(results_dir, "analysis")
    
    ensure_dir(save_dir)
    
    # Compute majority classifier baseline
    majority_accuracy = compute_majority_classifier_baseline(performance_matrix)
    
    all_results = []
    
    # Check each feature set
    for featureset in ['d2v', 'traditional', 'hybrid']:
        predictions_dir = os.path.join(results_dir, "predictions", featureset)
        
        if not os.path.exists(predictions_dir):
            log_info(f"Predictions directory not found: {predictions_dir}")
            continue
        
        log_info(f"{featureset.upper()} Feature Set - Top-1 Accuracy:")
        
        # Get all repetition files
        rep_files = sorted([f for f in os.listdir(predictions_dir) if f.startswith("rep_")])
        
        for rep_file in rep_files:
            rep_num = int(rep_file.split("_")[1].split(".")[0])
            df = pd.read_csv(os.path.join(predictions_dir, rep_file))
            
            # Extract algorithm names from column names
            algo_cols = [col.replace("_actual", "").replace("_pred", "") 
                         for col in df.columns if "_actual" in col]
            algo_cols = sorted(set(algo_cols))
            
            # For each sample, get top actual and top predicted algorithms
            correct_predictions = 0
            total_samples = len(df)
            
            for idx, row in df.iterrows():
                actual_values = np.array([row[f"{algo}_actual"] for algo in algo_cols])
                pred_values = np.array([row[f"{algo}_pred"] for algo in algo_cols])
                
                top_actual = get_top_algorithm(actual_values, algo_cols)
                top_pred = get_top_algorithm(pred_values, algo_cols)
                
                if top_actual == top_pred:
                    correct_predictions += 1
            
            top_1_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            
            all_results.append({
                'repetition': rep_num,
                'featureset': featureset,
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'top_1_accuracy': top_1_accuracy,
                'majority_baseline': majority_accuracy
            })
            
            log_info(f"  Rep {rep_num}: Top-1 Accuracy = {top_1_accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    # Combine all results
    combined_df = pd.DataFrame(all_results)
    combined_df.to_csv(os.path.join(save_dir, "top1_accuracy_analysis.csv"), index=False)
    
    # Summary
    print("\n" + "="*70)
    print("TOP-1 ACCURACY SUMMARY")
    print("="*70)
    
    for featureset in ['d2v', 'traditional', 'hybrid']:
        fs_data = combined_df[combined_df['featureset'] == featureset]
        if len(fs_data) > 0:
            mean_acc = fs_data['top_1_accuracy'].mean()
            std_acc = fs_data['top_1_accuracy'].std()
            print(f"\n{featureset.upper()}:")
            print(f"  Mean Top-1 Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  vs Majority Baseline: {majority_accuracy:.4f}")
            print(f"  Improvement: {mean_acc - majority_accuracy:+.4f}")
    
    return combined_df


def run_meta_regressor_multioutput(
    metric_name: str = "accuracy",
    algorithms: Optional[List[str]] = None,
    n_repeats: int = 10,
    seed: int = 175,
    output_dir: Optional[str] = None
) -> Tuple[Dict, float, pd.DataFrame]:
    """Run the complete meta-regressor pipeline with multi-output regression."""
    if metric_name not in ["accuracy", "f1"]:
        raise ValueError(f"metric_name must be 'accuracy' or 'f1', got '{metric_name}'")

    metatarget_path = os.path.join("meta_targets", f"{metric_name}.csv")

    if output_dir is None:
        algo_suffix = "_".join(sorted(algorithms)) if algorithms else "all_algorithms"
        output_dir = os.path.join("meta_regressor_results", f"{metric_name}_{algo_suffix}")

    ensure_dir(output_dir)

    # ========================================================================
    # SELF-HEALING: Fill missing accuracies if metric_name == "accuracy"
    # ========================================================================
    if metric_name == "accuracy":
        log_info("Running self-healing accuracy filler...")
        fill_missing_accuracies(metatarget_path, data_root="test_datasets")

    # Load data
    performance_matrix = load_metatargets(metatarget_path, metric_name)
    
    # Filter algorithms if specified
    if algorithms:
        missing_algos = set(algorithms) - set(performance_matrix.columns)
        if missing_algos:
            raise ValueError(f"Algorithms not found in metatarget: {missing_algos}")
        performance_matrix = performance_matrix[algorithms]
        log_info(f"Filtered to algorithms: {algorithms}")
    
    d2v, traditional, hybrid = load_metafeatures()

    metafeature_dict = {'d2v': d2v, 'traditional': traditional, 'hybrid': hybrid}
    datasets = align_features_with_targets(metafeature_dict, performance_matrix)

    # Compute baseline
    baseline_mae = compute_baseline(performance_matrix)

    # Run evaluation (saves all predictions internally)
    results = run_regression_evaluation(
        datasets,
        n_repeats=n_repeats,
        seed=seed,
        save_dir=output_dir
    )

    # Summarize results
    summary_df = summarize_results(results)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    # Analyze top-1 accuracy vs majority classifier
    analyze_top1_predictions(output_dir, performance_matrix)

    # Display summary
    print("\n📊 Summary of Meta-Regressor Performance:")
    print(summary_df.to_string(index=False))

    log_info(f"Results saved to: {output_dir}/")

    return results, baseline_mae, summary_df


if __name__ == "__main__":
    results, baseline_mae, summary_df = run_meta_regressor_multioutput()