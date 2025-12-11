#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metafeatures_local.py
---------------------
Extracts OpenML "qualities" (traditional meta-features) for datasets
stored locally in `test_datasets/` (folders named either `6`, `12`, or `dataset_6`).

Outputs:
  - qualities/traditional/metafeatures_raw.csv
  - qualities/traditional/metafeatures.csv
"""

import os
import openml
import pandas as pd
from sklearn.impute import KNNImputer


def analyze_and_impute_missing_data(df, output_dir):
    """Impute missing values using KNN and save processed file as metafeatures.csv."""
    if df.empty or "dataset_id" not in df.columns:
        print("⚠️ No data available for imputation.")
        return df

    features_only = df.drop(columns=["dataset_id"])
    total_missing = features_only.isnull().sum().sum()
    total_cells = features_only.size
    missing_pct = total_missing / total_cells * 100
    print(f"ℹ️ Missing data before imputation: {missing_pct:.2f}%")

    # Drop fully empty columns
    features_cleaned = features_only.dropna(axis=1, how="all")

    imputer = KNNImputer(n_neighbors=5)
    features_imputed = imputer.fit_transform(features_cleaned)

    df_imputed = pd.DataFrame(features_imputed, columns=features_cleaned.columns)
    df_imputed.insert(0, "dataset_id", df["dataset_id"])
    processed_path = os.path.join(output_dir, "metafeatures.csv")
    df_imputed.to_csv(processed_path, index=False)

    print(f"✅ Processed metafeatures saved to: {processed_path} "
          f"({df_imputed.shape[0]} rows, {df_imputed.shape[1]} columns)")
    return df_imputed


def extract_metafeatures_from_local_datasets(
    data_root="test_datasets",
    output_dir="qualities/traditional",
    impute=True
):
    """
    Extract OpenML meta-features (qualities) for locally stored datasets.
    Accepts folders named either `dataset_<id>` or `<id>`.
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_folders = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
        and (name.startswith("dataset_") or name.isdigit())
    ]

    if not dataset_folders:
        print(f"⚠️ No valid dataset folders found in {data_root}. "
              f"Expected names like 'dataset_6' or '6'.")
        return pd.DataFrame()

    dataset_ids = []
    for name in dataset_folders:
        try:
            dataset_ids.append(int(name.replace("dataset_", "")) if name.startswith("dataset_") else int(name))
        except ValueError:
            continue

    print(f"🔍 Found {len(dataset_ids)} datasets: {dataset_ids}")

    quality_list = openml.datasets.list_qualities()
    rows = []
    for dataset_id in sorted(dataset_ids):
        try:
            dataset = openml.datasets.get_dataset(dataset_id, download_qualities=True)
            qualities = dataset.qualities
            row = {"dataset_id": dataset_id}
            row.update({q: qualities.get(q) for q in quality_list})
            rows.append(row)
        except Exception as e:
            print(f"⚠️ Skipped dataset {dataset_id}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No metafeatures retrieved. Nothing saved.")
        return df

    raw_path = os.path.join(output_dir, "metafeatures_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"📁 Raw metafeatures saved to: {raw_path} "
          f"({df.shape[0]} rows, {df.shape[1]} columns)")

    if impute:
        return analyze_and_impute_missing_data(df, output_dir)
    else:
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract OpenML metafeatures for locally stored datasets.")
    parser.add_argument("--data_root", type=str, default="test_datasets", help="Path to local dataset folders.")
    parser.add_argument("--impute", action="store_true", help="Perform KNN imputation.")
    args = parser.parse_args()

    extract_metafeatures_from_local_datasets(args.data_root, impute=args.impute)
