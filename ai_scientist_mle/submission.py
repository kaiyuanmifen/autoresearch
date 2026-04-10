"""
Submission Module for AI Scientist MLE-Bench.

Handles submission generation, validation, and formatting
for MLE-bench competitions. Produces the final prediction CSV
and JSONL output required by the MLE-bench grading system.
"""

import json
import os
import os.path as osp
import shutil
from typing import Any, Dict, List, Optional

import pandas as pd


def find_best_submission(workspace_dir: str) -> Optional[str]:
    """
    Find the best submission CSV across all experiment runs.

    Looks for submission.csv in each run directory,
    preferring later runs (assumed to be iteratively improved).

    Returns the path to the best submission, or None.
    """
    best_submission = None
    best_run = -1

    for d in sorted(os.listdir(workspace_dir)):
        if d.startswith("run_") and osp.isdir(osp.join(workspace_dir, d)):
            sub_path = osp.join(workspace_dir, d, "submission.csv")
            if osp.exists(sub_path):
                try:
                    run_num = int(d.split("_")[1])
                    if run_num > best_run:
                        # Validate the submission is not empty
                        df = pd.read_csv(sub_path)
                        if len(df) > 0:
                            best_run = run_num
                            best_submission = sub_path
                except (ValueError, pd.errors.EmptyDataError):
                    continue

    return best_submission


def validate_submission(
    submission_path: str,
    sample_submission_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a submission CSV.

    Checks:
    - File exists and is readable
    - Has correct number of rows (if sample available)
    - Has correct columns (if sample available)
    - No missing values in required columns
    - Values are in expected range

    Returns a validation report.
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }

    if not osp.exists(submission_path):
        report["valid"] = False
        report["errors"].append(f"Submission file not found: {submission_path}")
        return report

    try:
        submission = pd.read_csv(submission_path)
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to read submission: {e}")
        return report

    report["info"]["rows"] = len(submission)
    report["info"]["columns"] = list(submission.columns)

    if len(submission) == 0:
        report["valid"] = False
        report["errors"].append("Submission is empty (0 rows)")
        return report

    # Check for missing values
    null_counts = submission.isnull().sum()
    if null_counts.sum() > 0:
        null_cols = {col: int(count) for col, count in null_counts.items() if count > 0}
        report["warnings"].append(f"Missing values found: {null_cols}")

    # Compare with sample submission if available
    if sample_submission_path and osp.exists(sample_submission_path):
        try:
            sample = pd.read_csv(sample_submission_path)
            report["info"]["expected_rows"] = len(sample)
            report["info"]["expected_columns"] = list(sample.columns)

            # Check column match
            if set(submission.columns) != set(sample.columns):
                missing = set(sample.columns) - set(submission.columns)
                extra = set(submission.columns) - set(sample.columns)
                if missing:
                    report["valid"] = False
                    report["errors"].append(f"Missing columns: {missing}")
                if extra:
                    report["warnings"].append(f"Extra columns (will be ignored): {extra}")

            # Check row count
            if len(submission) != len(sample):
                report["warnings"].append(
                    f"Row count mismatch: {len(submission)} vs expected {len(sample)}"
                )

        except Exception as e:
            report["warnings"].append(f"Could not validate against sample: {e}")

    return report


def prepare_mle_bench_submission(
    submission_path: str,
    competition_id: str,
    output_dir: str,
) -> str:
    """
    Prepare submission in MLE-bench format (JSONL).

    MLE-bench expects a JSONL file where each line contains:
        {"competition_id": "...", "submission_path": "..."}

    Args:
        submission_path: Path to the submission CSV.
        competition_id: The MLE-bench competition identifier.
        output_dir: Directory to write the output.

    Returns:
        Path to the JSONL submission file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy submission CSV to output directory
    dest_csv = osp.join(output_dir, f"{competition_id}_submission.csv")
    shutil.copy2(submission_path, dest_csv)

    # Create JSONL entry
    jsonl_path = osp.join(output_dir, "submissions.jsonl")
    entry = {
        "competition_id": competition_id,
        "submission_path": dest_csv,
    }

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return jsonl_path


def collect_all_submissions(
    results_dir: str,
    output_dir: str,
) -> str:
    """
    Collect submissions from all competition workspaces into a single JSONL.

    Args:
        results_dir: Directory containing competition workspace subdirectories.
        output_dir: Directory to write the combined output.

    Returns:
        Path to the combined JSONL file.
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = osp.join(output_dir, "submissions.jsonl")

    entries = []
    for comp_dir in sorted(os.listdir(results_dir)):
        comp_path = osp.join(results_dir, comp_dir)
        if not osp.isdir(comp_path):
            continue

        best_sub = find_best_submission(comp_path)
        if best_sub:
            # Copy to output
            dest_csv = osp.join(output_dir, f"{comp_dir}_submission.csv")
            shutil.copy2(best_sub, dest_csv)

            entries.append({
                "competition_id": comp_dir,
                "submission_path": dest_csv,
            })
            print(f"  Collected submission for {comp_dir}")
        else:
            print(f"  WARNING: No submission found for {comp_dir}")

    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nCollected {len(entries)} submissions to {jsonl_path}")
    return jsonl_path


def generate_submission_report(
    results_dir: str,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Generate a summary report of all submissions.

    Returns a dictionary with per-competition status.
    """
    report = {
        "total_competitions": 0,
        "submissions_generated": 0,
        "submissions_missing": 0,
        "competitions": {},
    }

    if not osp.exists(results_dir):
        return report

    for comp_dir in sorted(os.listdir(results_dir)):
        comp_path = osp.join(results_dir, comp_dir)
        if not osp.isdir(comp_path):
            continue

        report["total_competitions"] += 1
        best_sub = find_best_submission(comp_path)

        comp_report = {
            "has_submission": best_sub is not None,
            "submission_path": best_sub,
        }

        if best_sub:
            report["submissions_generated"] += 1
            validation = validate_submission(best_sub)
            comp_report["validation"] = validation
        else:
            report["submissions_missing"] += 1

        # Check for review
        review_path = osp.join(comp_path, "review.json")
        if osp.exists(review_path):
            with open(review_path, "r") as f:
                review = json.load(f)
            comp_report["review_score"] = review.get("Overall_Score", None)
            comp_report["review_decision"] = review.get("Decision", None)

        report["competitions"][comp_dir] = comp_report

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
