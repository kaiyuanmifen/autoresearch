"""
Solution Review Module for AI Scientist MLE-Bench.

Reviews completed ML solutions, evaluates their quality, identifies
improvements, and provides structured feedback for iteration.

Analogous to the paper review phase in the original AI Scientist,
but focused on ML solution quality rather than paper quality.
"""

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai_scientist_mle.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)


REVIEWER_SYSTEM_PROMPT = """You are an expert ML competition reviewer and Kaggle Grandmaster.
You are reviewing a solution submitted for an ML competition.
Be critical but constructive. Focus on practical improvements that would
increase the competition score."""

REVIEW_PROMPT = """Review the following ML competition solution.

## Competition Analysis
{analysis}

## Approach Used
{approach}

## Solution Code
```python
{solution_code}
```

## Experiment Results
{results}

## Notes
{notes}

Evaluate this solution and provide a structured review.

Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, carefully analyze:
1. Is the data preprocessing appropriate?
2. Is the model choice suitable for the problem?
3. Is feature engineering adequate?
4. Are hyperparameters reasonable?
5. Is the code robust and efficient?
6. Are there obvious improvements?
7. Is the submission format correct?
8. Are there any bugs or issues?

In <JSON>, provide:
- "Summary": Brief summary of the solution
- "Strengths": List of strengths
- "Weaknesses": List of weaknesses
- "Code_Quality": Rating 1-5 (poor to excellent)
- "Approach_Suitability": Rating 1-5
- "Feature_Engineering": Rating 1-5
- "Model_Selection": Rating 1-5
- "Hyperparameter_Tuning": Rating 1-5
- "Robustness": Rating 1-5
- "Overall_Score": Rating 1-10
- "Suggested_Improvements": List of specific, actionable improvements ranked by expected impact
- "Critical_Issues": List of any bugs or critical problems
- "Estimated_Percentile": Estimated competition percentile (0-100, higher is better)
- "Decision": "iterate" (needs improvement) or "submit" (ready for final submission)
"""

REVIEW_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
Reconsider your review. Did you miss any important aspects?

Ensure your review is:
- Specific and actionable
- Focused on improvements that would most impact the score
- Realistic about what can be achieved

Respond in the same format:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to change, repeat the JSON and include "I am done" in your thoughts."""

IMPROVEMENT_PROMPT = """Based on the following review of your ML solution, make improvements.

## Review
{review}

## Priority Improvements
Focus on the top-impact improvements:
{improvements}

## Critical Issues (fix these first)
{critical_issues}

Please modify `solution.py` to address these issues.
Focus on the changes most likely to improve the competition score.
Maintain code robustness - don't break what already works.
"""


def load_solution_artifacts(workspace_dir: str) -> Dict[str, str]:
    """Load solution code, results, and notes from workspace."""
    artifacts = {
        "solution_code": "",
        "results": "No results available.",
        "notes": "",
    }

    # Load solution code
    solution_path = osp.join(workspace_dir, "solution.py")
    if osp.exists(solution_path):
        with open(solution_path, "r") as f:
            artifacts["solution_code"] = f.read()

    # Load latest results
    run_results = []
    for d in sorted(os.listdir(workspace_dir)):
        if d.startswith("run_") and osp.isdir(osp.join(workspace_dir, d)):
            info_path = osp.join(workspace_dir, d, "final_info.json")
            if osp.exists(info_path):
                with open(info_path, "r") as f:
                    run_results.append({d: json.load(f)})
    if run_results:
        artifacts["results"] = json.dumps(run_results, indent=2)

    # Load notes
    notes_path = osp.join(workspace_dir, "notes.txt")
    if osp.exists(notes_path):
        with open(notes_path, "r") as f:
            artifacts["notes"] = f.read()

    return artifacts


def perform_review(
    workspace_dir: str,
    analysis: Dict,
    approach: Dict,
    client: Any,
    model: str,
    num_reflections: int = 3,
    num_reviews_ensemble: int = 1,
    temperature: float = 0.5,
) -> Dict:
    """
    Review a completed ML solution.

    Args:
        workspace_dir: Path to the solution workspace.
        analysis: Competition analysis dictionary.
        approach: The approach that was used.
        client: LLM client.
        model: Model name.
        num_reflections: Number of review refinement rounds.
        num_reviews_ensemble: Number of ensemble reviews.
        temperature: Sampling temperature.

    Returns:
        Review dictionary with scores and improvement suggestions.
    """
    artifacts = load_solution_artifacts(workspace_dir)

    analysis_str = json.dumps(
        {k: v for k, v in analysis.items() if k not in ("raw_data", "competition_dir")},
        indent=2,
    )
    approach_str = json.dumps(approach, indent=2)

    review_prompt = REVIEW_PROMPT.format(
        analysis=analysis_str,
        approach=approach_str,
        solution_code=artifacts["solution_code"][:8000],  # Truncate long code
        results=artifacts["results"][:3000],
        notes=artifacts["notes"][:2000],
    )

    if num_reviews_ensemble > 1:
        # Ensemble reviewing
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            review_prompt,
            model=model,
            client=client,
            system_message=REVIEWER_SYSTEM_PROMPT,
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )

        parsed_reviews = []
        for idx, rev in enumerate(llm_reviews):
            try:
                parsed = extract_json_between_markers(rev)
                if parsed:
                    parsed_reviews.append(parsed)
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")

        if not parsed_reviews:
            print("All ensemble reviews failed. Falling back to single review.")
            return _single_review(review_prompt, client, model, num_reflections, temperature)

        # Aggregate scores
        review = parsed_reviews[0].copy()
        for score_key in (
            "Code_Quality", "Approach_Suitability", "Feature_Engineering",
            "Model_Selection", "Hyperparameter_Tuning", "Robustness",
            "Overall_Score", "Estimated_Percentile",
        ):
            scores = [r.get(score_key, 0) for r in parsed_reviews if score_key in r]
            if scores:
                review[score_key] = int(round(np.mean(scores)))

        # Merge improvements from all reviews
        all_improvements = []
        seen = set()
        for r in parsed_reviews:
            for imp in r.get("Suggested_Improvements", []):
                if imp not in seen:
                    all_improvements.append(imp)
                    seen.add(imp)
        review["Suggested_Improvements"] = all_improvements

        # Merge critical issues
        all_issues = []
        seen = set()
        for r in parsed_reviews:
            for issue in r.get("Critical_Issues", []):
                if issue not in seen:
                    all_issues.append(issue)
                    seen.add(issue)
        review["Critical_Issues"] = all_issues

    else:
        review = _single_review(review_prompt, client, model, num_reflections, temperature)

    # Save review
    review_path = osp.join(workspace_dir, "review.json")
    with open(review_path, "w") as f:
        json.dump(review, f, indent=2)

    return review


def _single_review(
    review_prompt: str,
    client: Any,
    model: str,
    num_reflections: int,
    temperature: float,
) -> Dict:
    """Perform a single review with optional reflections."""
    text, msg_history = get_response_from_llm(
        review_prompt,
        client=client,
        model=model,
        system_message=REVIEWER_SYSTEM_PROMPT,
        temperature=temperature,
    )

    review = extract_json_between_markers(text)
    if review is None:
        return {"Overall_Score": 0, "Decision": "iterate", "error": "Failed to parse review"}

    # Reflection rounds
    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                REVIEW_REFLECTION_PROMPT.format(
                    current_round=j + 2,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=REVIEWER_SYSTEM_PROMPT,
                msg_history=msg_history,
                temperature=temperature,
            )
            refined = extract_json_between_markers(text)
            if refined:
                review = refined
            if "I am done" in text:
                break

    return review


def perform_improvement(
    review: Dict,
    coder: Any,
) -> str:
    """
    Apply review-suggested improvements to the solution.

    Args:
        review: Review dictionary with suggestions.
        coder: An aider Coder instance.

    Returns:
        Coder output string.
    """
    improvements = review.get("Suggested_Improvements", [])
    critical_issues = review.get("Critical_Issues", [])

    if not improvements and not critical_issues:
        return "No improvements suggested."

    improvements_str = "\n".join(f"- {imp}" for imp in improvements[:5])
    issues_str = "\n".join(f"- {issue}" for issue in critical_issues) if critical_issues else "None"

    prompt = IMPROVEMENT_PROMPT.format(
        review=json.dumps(review, indent=2),
        improvements=improvements_str,
        critical_issues=issues_str,
    )

    return coder.run(prompt)


def should_continue_iterating(review: Dict, iteration: int, max_iterations: int = 3) -> bool:
    """Decide whether to continue the review-improve loop."""
    if iteration >= max_iterations:
        return False

    decision = review.get("Decision", "iterate")
    overall_score = review.get("Overall_Score", 0)
    critical_issues = review.get("Critical_Issues", [])

    # Stop if review says to submit and score is decent
    if decision == "submit" and overall_score >= 6:
        return False

    # Stop if there are no critical issues and score is good
    if not critical_issues and overall_score >= 7:
        return False

    return True
