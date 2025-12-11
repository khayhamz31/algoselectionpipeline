#!/usr/bin/env python
# coding: utf-8

import os
import json
import pandas as pd
import openml
from tqdm import tqdm
from collections import defaultdict


# === STAGE 1: DOWNLOAD ALL RUNS ===
def download_all_runs(mapping_path, raw_dir):
    """Download all OpenML runs for each dataset/task and save to runs/raw/dataset_<id>.csv"""
    os.makedirs(raw_dir, exist_ok=True)

    with open(mapping_path, "r") as f:
        id_task_map = json.load(f)

    for dataset_id, task_id in tqdm(id_task_map.items(), desc="[1/4] Downloading all runs"):
        output_path = os.path.join(raw_dir, f"dataset_{dataset_id}.csv")
        try:
            runs_df = openml.runs.list_runs(task=[int(task_id)], output_format="dataframe")
            runs_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"⚠️ Failed to fetch task {task_id}: {e}")


# === STAGE 2: FILTER AND SAMPLE ===
def filter_and_sample_runs(flow_map_path, raw_dir, sampled_dir, sample_size=50):
    """Filter raw runs by algorithm and sample fixed number per algorithm."""
    os.makedirs(sampled_dir, exist_ok=True)

    with open(flow_map_path, "r") as f:
        flow_map = json.load(f)

    valid_flow_ids = set(map(int, flow_map.keys()))
    flow_to_algo = {int(fid): entry["algorithm_type"] for fid, entry in flow_map.items()}

    for filename in tqdm(os.listdir(raw_dir), desc="[2/4] Filtering & sampling"):
        if not filename.endswith(".csv"):
            continue

        dataset_id = filename.replace("dataset_", "").replace(".csv", "")
        dataset_out = os.path.join(sampled_dir, f"dataset_{dataset_id}")
        os.makedirs(dataset_out, exist_ok=True)
        input_path = os.path.join(raw_dir, filename)

        try:
            runs_df = pd.read_csv(input_path)
            filtered_df = runs_df[runs_df["flow_id"].isin(valid_flow_ids)]
            if filtered_df.empty:
                continue

            algo_groups = defaultdict(list)
            for idx, row in filtered_df.iterrows():
                flow_id = row["flow_id"]
                algo = flow_to_algo.get(flow_id)
                if algo:
                    algo_groups[algo].append(idx)

            for algo, indices in algo_groups.items():
                k = min(sample_size, len(indices))
                sampled_df = filtered_df.loc[indices].sample(n=k, random_state=42)
                algo_name = algo.lower().replace(" ", "_")
                sampled_df.to_csv(os.path.join(dataset_out, f"{algo_name}.csv"), index=False)

        except Exception as e:
            print(f"❌ Error filtering {filename}: {e}")


# === STAGE 3: FETCH METRICS ===
def fetch_metrics(sampled_dir, acc_dir, f1_dir, batch_size=50):
    """Fetch predictive_accuracy and f_measure metrics for each sampled dataset/algorithm."""
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(f1_dir, exist_ok=True)

    for dataset_folder in tqdm(os.listdir(sampled_dir), desc="[3/4] Fetching metrics"):
        dataset_path = os.path.join(sampled_dir, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue

        acc_out_dir = os.path.join(acc_dir, dataset_folder)
        f1_out_dir = os.path.join(f1_dir, dataset_folder)
        os.makedirs(acc_out_dir, exist_ok=True)
        os.makedirs(f1_out_dir, exist_ok=True)

        algo_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]

        for algo_file in algo_files:
            algo_name = algo_file.replace(".csv", "")
            algo_path = os.path.join(dataset_path, algo_file)

            try:
                sampled_df = pd.read_csv(algo_path)
                run_ids = sampled_df["run_id"].tolist()
                accuracy_vals, f1_vals = [], []

                for j in range(0, len(run_ids), batch_size):
                    batch = run_ids[j:j + batch_size]
                    evals_acc = openml.evaluations.list_evaluations(
                        function="predictive_accuracy", runs=batch, output_format="dataframe"
                    )
                    if not evals_acc.empty:
                        accuracy_vals.extend(evals_acc["value"].tolist())

                    evals_f1 = openml.evaluations.list_evaluations(
                        function="f_measure", runs=batch, output_format="dataframe"
                    )
                    if not evals_f1.empty:
                        f1_vals.extend(evals_f1["value"].tolist())

                pd.DataFrame({"accuracy": accuracy_vals}).to_csv(
                    os.path.join(acc_out_dir, f"{algo_name}.csv"), index=False
                )
                pd.DataFrame({"f1": f1_vals}).to_csv(
                    os.path.join(f1_out_dir, f"{algo_name}.csv"), index=False
                )

            except Exception as e:
                print(f"⚠️ Failed to fetch metrics for {dataset_folder}/{algo_name}: {e}")


# === STAGE 4: BUILD METATARGETS ===
def build_metatargets(acc_dir, f1_dir, meta_dir):
    """Aggregate per-dataset accuracy and f1 results into metatarget CSVs."""
    os.makedirs(meta_dir, exist_ok=True)
    acc_meta, f1_meta = [], []

    for dataset_folder in tqdm(os.listdir(acc_dir), desc="[4/4] Building metatargets"):
        dataset_id = dataset_folder.replace("dataset_", "")
        dataset_acc_path = os.path.join(acc_dir, dataset_folder)
        dataset_f1_path = os.path.join(f1_dir, dataset_folder)

        acc_row, f1_row = {"dataset_id": dataset_id}, {"dataset_id": dataset_id}

        if os.path.exists(dataset_acc_path):
            for algo_file in os.listdir(dataset_acc_path):
                algo_name = algo_file.replace(".csv", "")
                acc_file = os.path.join(dataset_acc_path, algo_file)
                acc_vals = pd.read_csv(acc_file)["accuracy"].dropna()
                if len(acc_vals) > 0:
                    acc_row[algo_name] = acc_vals.sort_values(ascending=False).head(10).median()

        if os.path.exists(dataset_f1_path):
            for algo_file in os.listdir(dataset_f1_path):
                algo_name = algo_file.replace(".csv", "")
                f1_file = os.path.join(dataset_f1_path, algo_file)
                f1_vals = pd.read_csv(f1_file)["f1"].dropna()
                if len(f1_vals) > 0:
                    f1_row[algo_name] = f1_vals.sort_values(ascending=False).head(10).median()

        acc_meta.append(acc_row)
        f1_meta.append(f1_row)

    pd.DataFrame(acc_meta).to_csv(os.path.join(meta_dir, "accuracy.csv"), index=False)
    pd.DataFrame(f1_meta).to_csv(os.path.join(meta_dir, "f1.csv"), index=False)


# === MASTER WRAPPER ===
def run_pipeline(
    mapping_path="test_datasets/id_task_mapping.json",
    flow_map_path="flows/filtered_flow_algorithm_mapping_v2.json",
    sample_size=50,
    batch_size=50,
    base_dir="runs"
):
    """
    Run all stages of the OpenML data pipeline in order.
    Final outputs: accuracy.csv and f1.csv saved in ./meta_targets/
    """

    raw_dir = os.path.join(base_dir, "raw")
    sampled_dir = os.path.join(base_dir, "sampled")
    acc_dir = os.path.join(base_dir, "accuracies")
    f1_dir = os.path.join(base_dir, "f1")
    meta_dir = os.path.join("meta_targets")  # ✅ new folder for meta-targets

    download_all_runs(mapping_path, raw_dir)
    filter_and_sample_runs(flow_map_path, raw_dir, sampled_dir, sample_size)
    fetch_metrics(sampled_dir, acc_dir, f1_dir, batch_size)
    build_metatargets(acc_dir, f1_dir, meta_dir)
