"""
Writeup Module for AI Scientist MLE-Bench.

Generates a structured technical report of the ML competition solution,
documenting the approach, experiments, results, and findings.

Unlike the original AI Scientist which generates LaTeX papers,
this produces a Markdown report suitable for documenting the
competition solution methodology.
"""

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

from ai_scientist_mle.llm import get_response_from_llm, extract_json_between_markers


WRITEUP_SYSTEM_PROMPT = """You are an expert technical writer specializing in
machine learning competition solution writeups. Your reports are clear,
well-structured, and include all relevant details about the approach,
experiments, and results."""

WRITEUP_PROMPT = """Write a comprehensive technical report for the following ML competition solution.

## Competition
{competition_info}

## Approach
{approach}

## Experiment Results
{results}

## Review Feedback
{review}

## Solution Code Summary
{code_summary}

## Notes
{notes}

Write the report in Markdown format with the following sections:

# Solution Report: {competition_id}

## 1. Competition Overview
- Problem description
- Evaluation metric
- Data description

## 2. Approach
- Strategy and rationale
- Model selection justification
- Feature engineering decisions

## 3. Implementation Details
- Data preprocessing pipeline
- Model architecture/configuration
- Training procedure
- Key hyperparameters

## 4. Experiments and Results
- Description of each experiment run
- Results and metrics
- What worked and what didn't

## 5. Analysis
- Key insights from the experiments
- Comparison with baseline
- Error analysis

## 6. Potential Improvements
- What could be tried with more time/compute
- Alternative approaches

## 7. Conclusion
- Summary of findings
- Final score and ranking estimate

Be specific, data-driven, and honest about limitations.
Include actual numbers and results from the experiments.
"""


def generate_writeup(
    workspace_dir: str,
    analysis: Dict,
    approach: Dict,
    results: List[Dict],
    review: Optional[Dict],
    client: Any,
    model: str,
) -> str:
    """
    Generate a technical report for the competition solution.

    Args:
        workspace_dir: Path to the solution workspace.
        analysis: Competition analysis dictionary.
        approach: The approach that was used.
        results: List of experiment run results.
        review: Optional review dictionary.
        client: LLM client.
        model: Model name.

    Returns:
        The generated report as a Markdown string.
    """
    # Prepare inputs
    competition_info = json.dumps(
        {k: v for k, v in analysis.items() if k not in ("raw_data", "competition_dir")},
        indent=2,
    )

    # Load solution code
    solution_path = osp.join(workspace_dir, "solution.py")
    code_summary = ""
    if osp.exists(solution_path):
        with open(solution_path, "r") as f:
            code = f.read()
        # Truncate if too long
        if len(code) > 5000:
            code_summary = code[:2500] + "\n\n... [truncated] ...\n\n" + code[-2500:]
        else:
            code_summary = code

    # Load notes
    notes = ""
    notes_path = osp.join(workspace_dir, "notes.txt")
    if osp.exists(notes_path):
        with open(notes_path, "r") as f:
            notes = f.read()

    prompt = WRITEUP_PROMPT.format(
        competition_info=competition_info,
        approach=json.dumps(approach, indent=2),
        results=json.dumps(results, indent=2) if results else "No results available.",
        review=json.dumps(review, indent=2) if review else "No review available.",
        code_summary=code_summary[:5000],
        notes=notes[:3000],
        competition_id=analysis.get("competition_id", "unknown"),
    )

    text, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message=WRITEUP_SYSTEM_PROMPT,
        temperature=0.3,
    )

    # Save the report
    report_path = osp.join(workspace_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(text)

    print(f"Report saved to {report_path}")
    return text


def generate_summary_report(
    all_competition_results: Dict[str, Dict],
    output_path: str,
) -> str:
    """
    Generate a summary report across all competitions.

    Args:
        all_competition_results: Dict mapping competition_id to results.
        output_path: Path to save the summary report.

    Returns:
        The summary report as a string.
    """
    lines = [
        "# MLE-Bench Results Summary",
        "",
        f"Total competitions attempted: {len(all_competition_results)}",
        "",
        "## Per-Competition Results",
        "",
        "| Competition | Status | Approach | Review Score | Submission |",
        "|------------|--------|----------|-------------|------------|",
    ]

    completed = 0
    for comp_id, result in sorted(all_competition_results.items()):
        status = "Completed" if result.get("success") else "Failed"
        approach_name = result.get("approach", {}).get("Name", "N/A")
        review_score = result.get("review", {}).get("Overall_Score", "N/A")
        has_submission = "Yes" if result.get("has_submission") else "No"

        if result.get("success"):
            completed += 1

        lines.append(
            f"| {comp_id} | {status} | {approach_name} | {review_score} | {has_submission} |"
        )

    lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- Completed: {completed}/{len(all_competition_results)}",
        f"- Completion rate: {completed / max(len(all_competition_results), 1) * 100:.1f}%",
    ])

    # Calculate average review score
    scores = [
        r.get("review", {}).get("Overall_Score", 0)
        for r in all_competition_results.values()
        if r.get("review", {}).get("Overall_Score")
    ]
    if scores:
        lines.append(f"- Average review score: {sum(scores) / len(scores):.1f}/10")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return report
