"""
Experiment Runner for AI Scientist MLE-Bench.

Implements and runs ML solutions for MLE-bench competitions.
Uses an LLM-driven coding agent (via aider) to write and iterate
on solution code, then executes experiments and captures results.

Adapted from the AI Scientist perform_experiments.py.
"""

import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from subprocess import TimeoutExpired
from typing import Any, Dict, List, Optional, Tuple

MAX_ITERS = 5  # Max retries per run on failure
MAX_RUNS = 5   # Max experiment runs
MAX_STDERR_OUTPUT = 1500
EXPERIMENT_TIMEOUT = 3600  # 1 hour default per experiment run

SOLUTION_PROMPT = """You are implementing a solution for an ML competition.

## Competition Analysis
{analysis}

## Selected Approach
{approach}

## Competition Directory
The competition data is in: {competition_dir}
Available files: {available_files}

## Instructions
Your goal is to implement the solution in `solution.py`.
The solution must:
1. Read training data from the competition directory
2. Train a model using the selected approach
3. Generate predictions on the test data
4. Save predictions to `submission.csv` in the expected format

The submission format should match:
{submission_format}

You have up to {max_runs} experiment runs. After each run, we'll evaluate
the results and you can refine your approach.

{baseline_info}

Plan your experiments:
1. First run: Implement the core solution
2. Subsequent runs: Iterate and improve based on results

After each change, we will run: `python solution.py --data_dir={competition_dir} --out_dir=run_i`
YOUR SOLUTION MUST ACCEPT THESE COMMAND LINE ARGS.

Important constraints:
- The solution must be self-contained in solution.py
- Use only standard ML libraries (sklearn, xgboost, lightgbm, pytorch, tensorflow, pandas, numpy)
- Handle missing values and edge cases robustly
- Set random seeds for reproducibility
- Print progress and metrics during training
"""

RESULTS_PROMPT = """Run {run_num} completed. Here are the results:
{results}

{score_info}

Decide if you need to modify your approach:
- If the submission was generated correctly, consider ways to improve the score
- If there were errors, fix them
- If the score is already good, consider if minor tweaks could push it higher

Please update `notes.txt` with:
- What you tried in run {run_num}
- The results obtained
- What you plan to try next

Then implement your next improvement in `solution.py`.
We will run: `python solution.py --data_dir={competition_dir} --out_dir=run_{next_run}`
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

If you are satisfied with the results and no further improvement is needed,
respond with 'ALL_COMPLETED'.
"""

ERROR_PROMPT = """Run {run_num} failed with the following error:
{error}

Please fix the issue in `solution.py` and we'll try again.
Common issues to check:
- File paths and data loading
- Missing value handling
- Column name mismatches
- Memory issues (try reducing data or model size)
- Import errors (use standard libraries only)
"""


def create_solution_workspace(
    approach: Dict,
    analysis: Dict,
    workspace_dir: str,
) -> str:
    """
    Create a workspace directory for the solution.

    Returns the path to the workspace.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    # Create initial solution.py template
    solution_template = '''"""
ML Competition Solution
Approach: {title}
Competition: {competition_id}
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main(data_dir: str, out_dir: str):
    """Main solution function."""
    os.makedirs(out_dir, exist_ok=True)

    # TODO: Implement solution
    # 1. Load data from data_dir
    # 2. Preprocess data
    # 3. Train model
    # 4. Generate predictions
    # 5. Save submission.csv and results

    print("Loading data...")
    # train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    # test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    print("Training model...")
    # TODO: Implement model training

    print("Generating predictions...")
    # TODO: Generate predictions

    # Save submission
    # submission = pd.DataFrame({{"id": test_ids, "target": predictions}})
    # submission.to_csv(os.path.join(out_dir, "submission.csv"), index=False)

    # Save run info
    info = {{
        "approach": "{name}",
        "status": "completed",
    }}
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
'''.format(
        title=approach.get("Title", "Unknown"),
        competition_id=analysis.get("competition_id", "unknown"),
        name=approach.get("Name", "unknown"),
    )

    solution_path = osp.join(workspace_dir, "solution.py")
    with open(solution_path, "w") as f:
        f.write(solution_template)

    # Create notes.txt
    notes_path = osp.join(workspace_dir, "notes.txt")
    with open(notes_path, "w") as f:
        f.write(f"# Competition: {analysis.get('competition_id', 'unknown')}\n")
        f.write(f"# Approach: {approach.get('Title', 'unknown')}\n")
        f.write(f"# Strategy: {approach.get('Strategy', '')}\n\n")
        f.write("## Run Log\n\n")

    return workspace_dir


def run_experiment(
    workspace_dir: str,
    competition_dir: str,
    run_num: int,
    timeout: int = EXPERIMENT_TIMEOUT,
) -> Tuple[int, str, Optional[Dict]]:
    """
    Execute a solution run.

    Returns:
        Tuple of (return_code, next_prompt, results_dict)
    """
    cwd = osp.abspath(workspace_dir)

    # Save a copy of the solution for this run
    solution_path = osp.join(workspace_dir, "solution.py")
    if osp.exists(solution_path):
        shutil.copy(solution_path, osp.join(workspace_dir, f"run_{run_num}.py"))

    out_dir = osp.join(workspace_dir, f"run_{run_num}")
    command = [
        "python", "solution.py",
        f"--data_dir={competition_dir}",
        f"--out_dir={out_dir}",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )

        stdout_output = result.stdout
        stderr_output = result.stderr

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(out_dir):
                shutil.rmtree(out_dir)

            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]

            next_prompt = ERROR_PROMPT.format(
                run_num=run_num,
                error=stderr_output,
            )
            return result.returncode, next_prompt, None

        # Check for results
        results = {}
        info_path = osp.join(out_dir, "final_info.json")
        if osp.exists(info_path):
            with open(info_path, "r") as f:
                results = json.load(f)

        # Check for submission
        submission_path = osp.join(out_dir, "submission.csv")
        submission_exists = osp.exists(submission_path)

        score_info = ""
        if submission_exists:
            try:
                import pandas as pd
                sub_df = pd.read_csv(submission_path)
                score_info = f"Submission generated: {sub_df.shape[0]} rows, columns: {list(sub_df.columns)}"
                results["submission_shape"] = list(sub_df.shape)
                results["submission_columns"] = list(sub_df.columns)
            except Exception as e:
                score_info = f"Submission file exists but could not be read: {e}"
        else:
            score_info = "WARNING: No submission.csv was generated!"

        # Include stdout info
        if stdout_output:
            # Truncate long output
            if len(stdout_output) > 3000:
                stdout_output = stdout_output[:1500] + "\n...\n" + stdout_output[-1500:]
            score_info += f"\n\nStdout output:\n{stdout_output}"

        next_prompt = RESULTS_PROMPT.format(
            run_num=run_num,
            results=json.dumps(results, indent=2),
            score_info=score_info,
            competition_dir=competition_dir,
            next_run=run_num + 1,
        )

        return result.returncode, next_prompt, results

    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(out_dir):
            shutil.rmtree(out_dir)
        next_prompt = f"""Run {run_num} timed out after {timeout} seconds.
Please optimize your solution for speed:
- Reduce model complexity or training iterations
- Use a smaller subset of data for initial development
- Optimize data loading and preprocessing
Then we'll try again."""
        return 1, next_prompt, None


def perform_experiments(
    approach: Dict,
    analysis: Dict,
    workspace_dir: str,
    competition_dir: str,
    coder: Any,
    timeout: int = EXPERIMENT_TIMEOUT,
) -> Tuple[bool, List[Dict]]:
    """
    Perform the full experiment loop for a given approach.

    This is the core experiment execution loop, analogous to
    AI Scientist's perform_experiments function.

    Args:
        approach: The selected approach dictionary.
        analysis: Competition analysis dictionary.
        workspace_dir: Path to the workspace directory.
        competition_dir: Path to competition data.
        coder: An aider Coder instance for code modification.
        timeout: Timeout per experiment run in seconds.

    Returns:
        Tuple of (success, list_of_run_results)
    """
    # List available files
    available_files = []
    if osp.exists(competition_dir):
        for f in os.listdir(competition_dir):
            fpath = osp.join(competition_dir, f)
            if osp.isfile(fpath):
                size = osp.getsize(fpath)
                available_files.append(f"{f} ({_format_size(size)})")

    # Prepare analysis string (without raw_data to save tokens)
    analysis_str = json.dumps(
        {k: v for k, v in analysis.items() if k not in ("raw_data", "competition_dir")},
        indent=2,
    )

    # Initial prompt to set up the solution
    initial_prompt = SOLUTION_PROMPT.format(
        analysis=analysis_str,
        approach=json.dumps(approach, indent=2),
        competition_dir=competition_dir,
        available_files="\n".join(available_files),
        submission_format=analysis.get("submission_format", "CSV with appropriate columns"),
        max_runs=MAX_RUNS,
        baseline_info="",
    )

    current_iter = 0
    run = 1
    all_results = []
    next_prompt = initial_prompt

    while run <= MAX_RUNS:
        if current_iter >= MAX_ITERS:
            print(f"Max iterations ({MAX_ITERS}) reached for run {run}")
            break

        # Let the coder modify the solution
        coder_out = coder.run(next_prompt)
        print(coder_out)

        if "ALL_COMPLETED" in coder_out:
            print("Coder indicated all experiments completed.")
            break

        # Run the experiment
        return_code, next_prompt, results = run_experiment(
            workspace_dir, competition_dir, run, timeout=timeout,
        )

        if return_code == 0:
            if results:
                all_results.append({"run": run, **results})
            run += 1
            current_iter = 0
        else:
            current_iter += 1

    if current_iter >= MAX_ITERS:
        print("Not all experiments completed due to repeated failures.")
        return False, all_results

    # Update notes with final summary
    _update_final_notes(workspace_dir, approach, all_results, coder)

    return True, all_results


def _update_final_notes(
    workspace_dir: str,
    approach: Dict,
    all_results: List[Dict],
    coder: Any,
):
    """Ask the coder to write final notes summarizing all runs."""
    notes_prompt = f"""Please update `notes.txt` with a comprehensive summary of all experiment runs.

Include:
- The approach used: {approach.get('Title', 'unknown')}
- Results from each run
- What worked and what didn't
- The best performing configuration
- Suggestions for further improvement

Run results: {json.dumps(all_results, indent=2)}
"""
    try:
        coder.run(notes_prompt)
    except Exception as e:
        print(f"Failed to update notes: {e}")


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
