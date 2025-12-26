from pathlib import Path
from typing import Optional
import shutil
import json
from datetime import datetime


def consolidate_run_outputs(
    selected_suite_info: dict,
    random_seed: int,
    root: Optional[Path] = None,
):
    if root is None:
        root = Path.cwd()

    suite_id = selected_suite_info["id"]
    suite_alias = selected_suite_info.get("alias", "unknown")

    run_dir = root / f"suite_{suite_id}_seed_{random_seed}"
    run_dir.mkdir(exist_ok=True)

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    folder_map = {
        "log": "logs",
        "meta_classifier_results": "results/meta_classifier_results",
        "meta_regressor_results": "results/meta_regressor_results",
        "regresscomparison": "results/regresscomparison",
        "meta_targets": "meta_targets",
        "qualities": "qualities",
        "runs": "runs",
        "test_datasets": "test_datasets",
    }

    moved, missing, skipped = [], [], []

    for src_name, dst_rel in folder_map.items():
        src = root / src_name
        dst = run_dir / dst_rel

        if not src.exists():
            missing.append(src_name)
            continue

        if dst.exists():
            skipped.append(src_name)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(src_name)

    run_info = {
        "suite_id": suite_id,
        "suite_alias": suite_alias,
        "random_seed": random_seed,
        "created_at": datetime.utcnow().isoformat(),
        "moved_folders": moved,
        "missing_folders": missing,
        "skipped_folders": skipped,
    }

    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("Moved:", moved)
    print("Missing:", missing)
    print("Skipped:", skipped)

    return run_dir
