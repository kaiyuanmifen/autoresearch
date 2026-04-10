"""
Competition Analysis Module for AI Scientist MLE-Bench.

Parses MLE-bench competition descriptions, data formats, evaluation metrics,
and constraints to provide structured context for solution generation.
"""

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

import pandas as pd

from ai_scientist_mle.llm import get_response_from_llm, extract_json_between_markers


ANALYSIS_SYSTEM_PROMPT = """You are an expert competitive machine learning engineer.
Your task is to analyze a Kaggle-style ML competition and produce a structured analysis
that will guide the development of a high-performing solution.

Be thorough, precise, and practical. Focus on actionable insights."""

ANALYSIS_PROMPT = """Analyze the following ML competition and produce a structured analysis.

## Competition Description
{description}

## Data Description
{data_description}

## Evaluation Metric
{evaluation_metric}

## Sample Submission Format
{sample_submission}

## Available Files
{available_files}

Respond in the following format:

THOUGHT:
<THOUGHT>

ANALYSIS JSON:
```json
<JSON>
```

In <THOUGHT>, reason through:
1. What type of ML problem is this? (classification, regression, ranking, segmentation, etc.)
2. What are the key features and target variables?
3. What preprocessing will be needed?
4. What evaluation metric is used and what does it optimize for?
5. What are the main challenges and potential pitfalls?
6. What baseline approaches would work?
7. What advanced approaches could achieve top scores?

In <JSON>, provide the analysis with these fields:
- "problem_type": The type of ML problem (e.g., "binary_classification", "multiclass_classification", "regression", "multi_label", "image_classification", "object_detection", "nlp_classification", "time_series", "recommendation")
- "target_variable": The target column or prediction output
- "evaluation_metric": The metric used for evaluation (e.g., "auc", "accuracy", "rmse", "map@k")
- "metric_direction": "maximize" or "minimize"
- "input_features": Brief description of input features/data
- "data_format": Description of data format (tabular, image, text, etc.)
- "key_challenges": List of main challenges
- "preprocessing_steps": List of recommended preprocessing steps
- "baseline_approaches": List of simple baseline approaches
- "advanced_approaches": List of advanced approaches likely to score well
- "submission_format": Description of the expected submission format
- "estimated_difficulty": "low", "medium", or "high"
- "time_budget_recommendation": Suggested time allocation in hours for each phase
"""


def load_competition_data(competition_dir: str) -> Dict[str, str]:
    """Load competition metadata from a directory."""
    result = {
        "description": "",
        "data_description": "",
        "evaluation_metric": "",
        "sample_submission": "",
        "available_files": "",
    }

    # Try to load description
    desc_path = osp.join(competition_dir, "description.txt")
    if osp.exists(desc_path):
        with open(desc_path, "r") as f:
            result["description"] = f.read()

    # Try to load from a combined metadata file
    meta_path = osp.join(competition_dir, "metadata.json")
    if osp.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        result["description"] = meta.get("description", result["description"])
        result["data_description"] = meta.get("data_description", "")
        result["evaluation_metric"] = meta.get("evaluation_metric", "")

    # Check for sample submission
    sample_sub = osp.join(competition_dir, "sample_submission.csv")
    if osp.exists(sample_sub):
        try:
            df = pd.read_csv(sample_sub, nrows=5)
            result["sample_submission"] = (
                f"Columns: {list(df.columns)}\n"
                f"Shape: {df.shape}\n"
                f"Sample rows:\n{df.to_string()}"
            )
        except Exception as e:
            result["sample_submission"] = f"Error reading sample submission: {e}"

    # List available files
    if osp.exists(competition_dir):
        files = []
        for root, dirs, filenames in os.walk(competition_dir):
            for fn in filenames:
                rel_path = osp.relpath(osp.join(root, fn), competition_dir)
                size = osp.getsize(osp.join(root, fn))
                files.append(f"  {rel_path} ({_format_size(size)})")
            # Don't recurse too deep
            if root.count(os.sep) - competition_dir.count(os.sep) > 2:
                dirs.clear()
        result["available_files"] = "\n".join(files[:50])
        if len(files) > 50:
            result["available_files"] += f"\n  ... and {len(files) - 50} more files"

    return result


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def analyze_competition(
    competition_dir: str,
    client: Any,
    model: str,
    competition_id: Optional[str] = None,
) -> Dict:
    """
    Analyze a competition and return structured analysis.

    Args:
        competition_dir: Path to the competition data directory.
        client: LLM client.
        model: Model name.
        competition_id: Optional competition identifier.

    Returns:
        Dictionary with structured competition analysis.
    """
    comp_data = load_competition_data(competition_dir)

    if not comp_data["description"]:
        print(f"Warning: No description found for competition in {competition_dir}")

    text, _ = get_response_from_llm(
        ANALYSIS_PROMPT.format(**comp_data),
        client=client,
        model=model,
        system_message=ANALYSIS_SYSTEM_PROMPT,
    )

    analysis = extract_json_between_markers(text)
    if analysis is None:
        print("Failed to parse competition analysis. Using defaults.")
        analysis = {
            "problem_type": "unknown",
            "target_variable": "unknown",
            "evaluation_metric": "unknown",
            "metric_direction": "maximize",
            "input_features": comp_data.get("data_description", ""),
            "data_format": "tabular",
            "key_challenges": [],
            "preprocessing_steps": [],
            "baseline_approaches": ["simple baseline"],
            "advanced_approaches": [],
            "submission_format": "",
            "estimated_difficulty": "medium",
            "time_budget_recommendation": {},
        }

    analysis["competition_id"] = competition_id or osp.basename(competition_dir)
    analysis["competition_dir"] = competition_dir
    analysis["raw_data"] = comp_data

    return analysis


def inspect_data_files(
    competition_dir: str,
    client: Any,
    model: str,
) -> Dict[str, Any]:
    """
    Inspect the actual data files to understand their structure.

    Returns a summary of each data file found.
    """
    summaries = {}

    for fname in os.listdir(competition_dir):
        fpath = osp.join(competition_dir, fname)
        if not osp.isfile(fpath):
            continue

        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(fpath, nrows=100)
                summaries[fname] = {
                    "type": "csv",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "null_counts": df.isnull().sum().to_dict(),
                    "sample": df.head(3).to_dict(),
                }
            except Exception as e:
                summaries[fname] = {"type": "csv", "error": str(e)}

        elif fname.endswith(".json"):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    summaries[fname] = {
                        "type": "json",
                        "structure": "list",
                        "length": len(data),
                        "sample_keys": list(data[0].keys()) if data and isinstance(data[0], dict) else None,
                    }
                elif isinstance(data, dict):
                    summaries[fname] = {
                        "type": "json",
                        "structure": "dict",
                        "keys": list(data.keys())[:20],
                    }
            except Exception as e:
                summaries[fname] = {"type": "json", "error": str(e)}

        elif fname.endswith((".parquet", ".pq")):
            try:
                df = pd.read_parquet(fpath)
                summaries[fname] = {
                    "type": "parquet",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }
            except Exception as e:
                summaries[fname] = {"type": "parquet", "error": str(e)}

    return summaries
