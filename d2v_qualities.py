import os
import pandas as pd
import subprocess
from tqdm import tqdm


def extract_metafeatures_from_datasets(
    data_root="test_datasets",
    extraction_script="extract_meta_features.py",
    extracted_folder="extracted",
    output_file="qualities/d2v/metafeatures.csv"
):
    """Extract meta-features for each dataset and combine all results into one file."""
    extracted_count = 0
    skipped_count = 0

    if not os.path.exists(data_root):
        print(f"[ERROR] Folder '{data_root}' not found.")
        return None

    dataset_names = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]
    print(f"[INFO] Found {len(dataset_names)} datasets in '{data_root}'")

    for dataset_name in tqdm(dataset_names, desc="Extracting meta-features"):
        dataset_path = os.path.join(data_root, dataset_name)
        predictors_file = os.path.join(dataset_path, f"{dataset_name}_py.dat")
        labels_file = os.path.join(dataset_path, "labels_py.dat")

        if not (os.path.exists(predictors_file) and os.path.exists(labels_file)):
            skipped_count += 1
            continue

        try:
            subprocess.run(
                ["python", extraction_script, "--file", dataset_name],
                capture_output=False,
                text=True,
                check=True
            )
            extracted_count += 1
        except subprocess.CalledProcessError:
            skipped_count += 1
        except Exception:
            skipped_count += 1

    print(f"[SUMMARY] Extracted: {extracted_count}, Skipped: {skipped_count}")

    combined_df = combine_metafeature_files(extracted_folder, output_file)
    if combined_df is not None:
        print(f"[DONE] Combined meta-features saved to '{output_file}' ({combined_df.shape})")
    else:
        print("[WARNING] No meta-feature files found to combine.")

    return combined_df


def combine_metafeature_files(
    extracted_folder="extracted",
    output_file="qualities/d2v/metafeatures.csv"
):
    """Combine all extracted CSV meta-feature files into one DataFrame."""
    if not os.path.exists(extracted_folder):
        print(f"[ERROR] Folder '{extracted_folder}' not found.")
        return None

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dfs = []

    for file in os.listdir(extracted_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(extracted_folder, file)
            try:
                df = pd.read_csv(file_path, index_col=0)
                df.index = [os.path.splitext(file)[0]]
                dfs.append(df)
            except Exception:
                continue

    if not dfs:
        return None

    combined_df = pd.concat(dfs)
    combined_df.to_csv(output_file)
    return combined_df


def extract_metafeatures_for_suite(suite_id, data_root="test_datasets"):
    """Extract and combine meta-features for a specific OpenML suite."""
    output_file = f"qualities/d2v/suite_{suite_id}_metafeatures.csv"
    combined_df = extract_metafeatures_from_datasets(
        data_root=data_root,
        output_file=output_file
    )
    return combined_df


if __name__ == "__main__":
    extract_metafeatures_from_datasets()