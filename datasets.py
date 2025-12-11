"""
OpenML Dataset Downloader and Preprocessor

This module provides utilities to download datasets from OpenML, preprocess features,
handle missing values, encode labels, and detect task types (classification, regression, multilabel).

Main functions:
  - download_and_process_dataset(): Download and process a single dataset
  - download_benchmark_suite(): Download all datasets from an OpenML benchmark suite
  - download_datasets_from_df(): Download datasets specified in a DataFrame
"""

import os
import json
import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from tqdm import tqdm


# ========================================================================================
# FEATURE PREPROCESSING
# ========================================================================================

def preprocess_features(X, missing_threshold=0.7):
    """
    Handle missing values in features with safe thresholds.
    
    Args:
        X: Input features (DataFrame or array-like)
        missing_threshold: Drop columns with > this fraction of missing values (default: 0.7)
    
    Returns:
        X_final: Preprocessed features (DataFrame)
        samples_to_keep: Boolean mask of kept samples
        categorical_features: List of categorical column names
        numerical_features: List of numerical column names
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    missing_pct = X.isnull().sum() / len(X)
    features_to_keep = missing_pct <= missing_threshold
    X_cleaned = X.loc[:, features_to_keep]

    if X_cleaned.shape[1] == 0:
        # If everything dropped, return as-is to avoid crashes
        return X_cleaned.copy(), pd.Series(False, index=X.index), [], []

    sample_missing_pct = X_cleaned.isnull().sum(axis=1) / max(1, len(X_cleaned.columns))
    samples_to_keep = sample_missing_pct <= 0.5
    X_final = X_cleaned.loc[samples_to_keep].copy()

    categorical_features = X_final.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X_final.select_dtypes(exclude=['object', 'category']).columns.tolist()

    if X_final.isnull().sum().sum() > 0:
        if categorical_features:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_final[categorical_features] = cat_imputer.fit_transform(X_final[categorical_features])
        if numerical_features:
            n_neighbors = min(5, len(X_final) - 1)
            if n_neighbors > 0:
                num_imputer = KNNImputer(n_neighbors=n_neighbors)
                X_final[numerical_features] = num_imputer.fit_transform(X_final[numerical_features])
            else:
                num_imputer = SimpleImputer(strategy='median')
                X_final[numerical_features] = num_imputer.fit_transform(X_final[numerical_features])

    return X_final, samples_to_keep, categorical_features, numerical_features


def encode_and_scale_features(X, categorical_features, numerical_features):
    """
    One-hot encode categorical features and MinMax scale numeric features.
    
    Args:
        X: Input features (DataFrame)
        categorical_features: List of categorical column names
        numerical_features: List of numerical column names
    
    Returns:
        X_processed: Encoded and scaled features as dense numpy array
    """
    transformers = []
    if numerical_features:
        transformers.append(('num', MinMaxScaler(), numerical_features))
    if categorical_features:
        # Handle sklearn version differences (sparse_output introduced in 1.2)
        try:
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        except TypeError:
            ohe = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        transformers.append(('cat', ohe, categorical_features))

    if transformers:
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        X_processed = preprocessor.fit_transform(X)
        # Ensure array (ColumnTransformer can return np array already)
        X_processed = np.asarray(X_processed)
        return X_processed
    else:
        return X.values


# ========================================================================================
# TASK TYPE AND LABEL DETECTION
# ========================================================================================

def detect_task_type(y):
    """
    Infer task type from the target variable.
    
    Args:
        y: Target variable (Series, DataFrame, array-like, or None)
    
    Returns:
        str: One of 'none', 'classification', 'regression', or 'multilabel'
    """
    if y is None:
        return 'none'

    # Convert to pandas representation for inspection
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            return 'none'
        if y.shape[1] > 1:
            return 'multilabel'
        # Single-column DataFrame -> treat like Series
        y_series = y.iloc[:, 0]
    elif isinstance(y, pd.Series):
        y_series = y
    else:
        y_series = pd.Series(y)

    # If completely empty:
    if len(y_series) == 0:
        return 'none'

    # Heuristics:
    # - If dtype is numeric and unique values high relative to n -> regression
    n = len(y_series)
    nunique = y_series.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(y_series)

    # Use threshold: if >= 50% of values are unique AND numeric, likely regression
    if is_numeric and nunique >= max(5, 0.5 * n):
        return 'regression'
    else:
        return 'classification'


def encode_labels(y, task_type):
    """
    Encode labels according to task type.
    
    Args:
        y: Target variable
        task_type: One of 'classification', 'regression', 'multilabel', or 'none'
    
    Returns:
        y_encoded: Encoded labels (1D or 2D array, or None)
        meta: Dictionary with metadata (class counts, label mappings, etc.)
    """
    if task_type == 'none':
        return None, {}

    if isinstance(y, pd.DataFrame):
        y_df = y.copy()
    elif isinstance(y, pd.Series):
        y_df = y.to_frame(name=y.name if y.name is not None else 'target')
    else:
        y_df = pd.DataFrame({'target': y})

    meta = {}

    if task_type == 'classification':
        col = y_df.columns[0]
        y_col = y_df[col]
        le = LabelEncoder()
        y_enc = le.fit_transform(y_col.astype(str) if not pd.api.types.is_numeric_dtype(y_col) else y_col)
        meta['classes'] = int(np.unique(y_enc).size)
        meta['class_labels'] = le.classes_.tolist()
        return y_enc, meta

    if task_type == 'regression':
        col = y_df.columns[0]
        y_arr = pd.to_numeric(y_df[col], errors='coerce').to_numpy(dtype=float)
        return y_arr, meta

    if task_type == 'multilabel':
        # Ensure every column is numeric; if not, label-encode per column.
        y_work = y_df.copy()
        class_labels = {}
        for c in y_work.columns:
            if pd.api.types.is_numeric_dtype(y_work[c]):
                # keep as-is (assumed 0/1 or numeric labels)
                continue
            le = LabelEncoder()
            y_work[c] = le.fit_transform(y_work[c].astype(str))
            class_labels[c] = le.classes_.tolist()
        y_arr = y_work.to_numpy(dtype=float)
        meta['label_columns'] = y_work.columns.tolist()
        if class_labels:
            meta['per_label_class_labels'] = class_labels
        return y_arr, meta

    raise ValueError(f"Unsupported task_type: {task_type}")


# ========================================================================================
# HELPERS
# ========================================================================================

def _safe_default_target_attribute(dataset):
    """
    Safely extract target column names from an OpenML dataset object.
    
    Args:
        dataset: OpenML dataset object
    
    Returns:
        list: Target column names (possibly empty)
    """
    tgt = dataset.default_target_attribute
    if tgt is None:
        return []
    if isinstance(tgt, str):
        parts = [p.strip() for p in tgt.split(',') if p.strip()]
        return parts
    if isinstance(tgt, list):
        return tgt
    return []


# ========================================================================================
# CORE DOWNLOADERS
# ========================================================================================

def download_and_process_dataset(dataset_id, dataset_name=None, target=None, output_dir="test_datasets"):
    """
    Download a dataset from OpenML, preprocess features, encode labels, and save artifacts.
    
    Args:
        dataset_id (int): OpenML dataset ID
        dataset_name (str, optional): Custom name for the dataset
        target (str or list, optional): Target column(s). If None, uses default_target_attribute
        output_dir (str): Directory to save processed data (default: "test_datasets")
    
    Returns:
        tuple: (success: bool, preprocessing_info: dict or None)
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id)

        # Prefer provided dataset_name; else fetch & clean
        if dataset_name is None:
            dataset_name = dataset.name.replace(" ", "_").replace("/", "_")

        # Determine target columns
        default_targets = _safe_default_target_attribute(dataset)
        if target is None:
            target_cols = default_targets
        else:
            # Allow caller to force one or multiple targets
            target_cols = target if isinstance(target, list) else [target]

        # Get full X,y; OpenML can return X with target removed only if target is provided.
        if target_cols:
            X, y, _, _ = dataset.get_data(target=target_cols)
        else:
            Xy, _, _, _ = dataset.get_data()  # all columns
            # If no declared target, set y=None and keep all columns in X
            X = Xy
            y = None

        # If target columns are in X (when unspecified), pop them out if they exist
        if y is None and target_cols:
            # Attempt to split y out of X by columns
            present_targets = [c for c in target_cols if c in X.columns]
            if present_targets:
                y = X[present_targets].copy()
                X = X.drop(columns=present_targets)

        # Ensure types
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Infer task type from y
        task_type = detect_task_type(y)

        # Preprocess features
        X_processed, valid_samples_mask, categorical_features, numerical_features = preprocess_features(X)

        # Align y to kept samples (if any)
        y_processed = None
        if task_type != 'none':
            # y could be Series or DataFrame
            if isinstance(y, pd.Series):
                y_processed = y.loc[valid_samples_mask].copy()
            elif isinstance(y, pd.DataFrame):
                y_processed = y.loc[valid_samples_mask].copy()
            else:
                y_series = pd.Series(y)
                y_processed = y_series.loc[valid_samples_mask].copy()

        # If after preprocessing too few samples, skip
        if len(X_processed) < 10:
            print(f"Dataset {dataset_id}: Too few samples after preprocessing ({len(X_processed)})")
            return False, None

        # Transform features (encode + scale)
        X_final = encode_and_scale_features(X_processed, categorical_features, numerical_features)

        # Encode labels per task
        y_final, label_meta = encode_labels(y_processed, task_type) if task_type != 'none' else (None, {})

        # Save artifacts
        folder_path = os.path.join(output_dir, str(dataset_id))
        os.makedirs(folder_path, exist_ok=True)

        np.savetxt(os.path.join(folder_path, f"{dataset_id}_py.dat"), X_final, fmt="%.6f", delimiter=",")
        if y_final is not None:
            # y_final can be 1D or 2D
            if y_final.ndim == 1:
                np.savetxt(os.path.join(folder_path, "labels_py.dat"), y_final, fmt="%.10g")
            else:
                np.savetxt(os.path.join(folder_path, "labels_py.dat"), y_final, fmt="%.10g", delimiter=",")

        # Build preprocessing info
        preprocessing_info = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "target_columns": target_cols if target_cols else [],
            "original_shape_rows": int(dataset.qualities.get('NumberOfInstances', X.shape[0])) if hasattr(dataset, 'qualities') and isinstance(dataset.qualities, dict) else int(X.shape[0]),
            "original_shape_cols": int(X.shape[1] + (len(target_cols) if target_cols else 0)),
            "final_shape_rows": int(X_final.shape[0]),
            "final_shape_cols": int(X_final.shape[1]),
            "samples_removed": int(X.shape[0] - X_processed.shape[0]),
            "features_removed": int(X.shape[1] - X_processed.shape[1]),
            "categorical_features": int(len(categorical_features)),
            "numerical_features": int(len(numerical_features)),
        }
        preprocessing_info.update(label_meta)

        return True, preprocessing_info

    except Exception as e:
        print(f"Failed to download dataset {dataset_id} ({dataset_name}): {e}")
        return False, None


def download_benchmark_suite(suite_id, output_dir="test_datasets"):
    """
    Download and process all datasets in an OpenML benchmark suite.

    Args:
        suite_id (int): OpenML suite ID
        output_dir (str): Directory to save processed data (default: "test_datasets")

    Returns:
        dict: Statistics with keys 'successful', 'failed', 'total'
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)

    preprocessing_log = []
    id_name_map = {}
    id_task_map = {}  # ✅ new mapping

    try:
        suite = openml.study.get_suite(suite_id)
        print(f"Found {len(suite.data)} datasets in suite")

        successful = 0
        failed = 0

        # === Build dataset → task mapping from suite ===
        for dataset_id, task_id in zip(suite.data, suite.tasks):
            id_task_map[str(dataset_id)] = int(task_id)

        for dataset_id in tqdm(suite.data, desc="Downloading datasets"):
            try:
                dataset = openml.datasets.get_dataset(dataset_id)
                dataset_name = dataset.name.replace(" ", "_").replace("/", "_")

                # Determine targets (may be empty or comma-separated multi-target)
                target_cols = _safe_default_target_attribute(dataset)

                id_name_map[str(dataset_id)] = dataset_name  # keep old mapping too

                success, preprocessing_info = download_and_process_dataset(
                    dataset_id, dataset_name, target=target_cols if target_cols else None, output_dir=output_dir
                )

                if success:
                    successful += 1
                    preprocessing_log.append(preprocessing_info)
                else:
                    failed += 1

            except Exception as e:
                print(f"Failed to process dataset {dataset_id}: {e}")
                failed += 1

        # === Save preprocessing log ===
        if preprocessing_log:
            log_df = pd.DataFrame(preprocessing_log)
            log_path = f"log/suite_{suite_id}_preprocessing_log.csv"
            log_df.to_csv(log_path, index=False)
            print(f"Preprocessing log saved to {log_path}")

        # === Save dataset → name mapping ===
        map_path = os.path.join(output_dir, "id_name_mapping.json")
        with open(map_path, "w") as f:
            json.dump(id_name_map, f, indent=2)
        print(f"Saved ID-name mapping to {map_path}")

        # === Save dataset → task mapping ===
        task_map_path = os.path.join(output_dir, "id_task_mapping.json")
        with open(task_map_path, "w") as f:
            json.dump(id_task_map, f, indent=2)
        print(f"Saved ID-task mapping to {task_map_path}")

        return {"successful": successful, "failed": failed, "total": len(suite.data)}

    except Exception as e:
        print(f"Failed to get benchmark suite {suite_id}: {e}")
        return {"successful": 0, "failed": 0, "total": 0}


def download_datasets_from_df(df, output_dir="test_datasets"):
    """
    Download and process datasets specified in a DataFrame.
    
    Args:
        df (DataFrame): DataFrame with columns: dataset_id, dataset_name (optional), target (optional)
        output_dir (str): Directory to save processed data (default: "test_datasets")
    
    Returns:
        dict: Statistics with keys 'successful', 'failed', 'total'
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("log", exist_ok=True)

    preprocessing_log = []
    id_name_map = {}
    successful = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading datasets"):
        dataset_id = int(row['dataset_id'])
        dataset_name = row.get('dataset_name', None)
        target = row.get('target', None)
        if isinstance(target, float) and pd.isna(target):
            target = None

        id_name_map[str(dataset_id)] = dataset_name if dataset_name else str(dataset_id)

        success, preprocessing_info = download_and_process_dataset(
            dataset_id, dataset_name=dataset_name, target=target, output_dir=output_dir
        )
        if success:
            successful += 1
            preprocessing_log.append(preprocessing_info)
        else:
            failed += 1

    # Save log
    if preprocessing_log:
        log_df = pd.DataFrame(preprocessing_log)
        log_path = "log/custom_datasets_preprocessing_log.csv"
        log_df.to_csv(log_path, index=False)
        print(f"Preprocessing log saved to {log_path}")

    # Save id-name map
    map_path = os.path.join(output_dir, "id_name_mapping.json")
    with open(map_path, "w") as f:
        json.dump(id_name_map, f, indent=2)
    print(f"Saved ID-name mapping to {map_path}")

    return {"successful": successful, "failed": failed, "total": len(df)}


if __name__ == "__main__":
    print("OpenML Dataset Downloader Module")
    print("Import this module and use the following functions:")
    print("  - download_benchmark_suite(suite_id)")
    print("  - download_and_process_dataset(dataset_id)")
    print("  - download_datasets_from_df(df)")