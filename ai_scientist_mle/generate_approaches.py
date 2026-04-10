"""
Approach Generation Module for AI Scientist MLE-Bench.

Generates candidate solution approaches for MLE-bench competitions,
analogous to the idea generation phase in the original AI Scientist.
Each approach specifies a modeling strategy, preprocessing pipeline,
and expected implementation steps.
"""

import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional

from ai_scientist_mle.llm import (
    get_response_from_llm,
    extract_json_between_markers,
)


APPROACH_SYSTEM_PROMPT = """You are an expert competitive machine learning engineer
who consistently achieves top placements in Kaggle competitions.

Your goal is to generate practical, implementable solution approaches that
maximize the competition metric. Focus on approaches that:
1. Are feasible to implement within the time and compute constraints
2. Have a strong track record on similar problem types
3. Can be iteratively improved
4. Balance complexity with likelihood of success"""

APPROACH_FIRST_PROMPT = """Given the following competition analysis, generate a solution approach.

## Competition Analysis
{analysis}

## Previously Generated Approaches
'''
{prev_approaches}
'''

Generate the next solution approach for this competition.
The approach should be different from previous ones and should be feasible to implement.

Respond in the following format:

THOUGHT:
<THOUGHT>

APPROACH JSON:
```json
<JSON>
```

In <THOUGHT>, reason through:
1. What types of models are most suitable for this problem?
2. What preprocessing and feature engineering would help?
3. How does this approach differ from the ones already generated?
4. What are the risks and how to mitigate them?
5. What is the expected performance level?

In <JSON>, provide the approach with these fields:
- "Name": A short identifier (lowercase, underscores, no spaces). E.g. "xgboost_baseline"
- "Title": A descriptive title for the approach
- "Strategy": High-level description of the approach
- "Model": The primary model/algorithm to use
- "Preprocessing": List of preprocessing steps
- "Feature_Engineering": List of feature engineering steps
- "Hyperparameters": Key hyperparameters to tune
- "Implementation_Plan": Step-by-step implementation plan
- "Expected_Strengths": Why this approach should work well
- "Expected_Weaknesses": Potential limitations
- "Complexity": Rating 1-5 (1=simple baseline, 5=complex ensemble)
- "Expected_Score": Rough estimate of expected competition score
- "Time_Estimate_Hours": Estimated time to implement in hours

Be realistic and practical. This JSON will be automatically parsed.
You will have {num_reflections} rounds to refine, but you don't need to use them all.
"""

APPROACH_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
Carefully review the approach you just created. Consider:
- Is the implementation plan detailed enough to be directly coded?
- Are there any missing preprocessing or feature engineering steps?
- Is the complexity appropriate for the time constraints?
- Could the approach be simplified while maintaining effectiveness?

Refine and improve the approach.

Respond in the same format:
THOUGHT:
<THOUGHT>

APPROACH JSON:
```json
<JSON>
```

If there is nothing to improve, repeat the previous JSON EXACTLY and include
"I am done" at the end of your thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


def generate_approaches(
    analysis: Dict,
    client: Any,
    model: str,
    max_num_approaches: int = 5,
    num_reflections: int = 3,
    seed_approaches: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Generate candidate solution approaches for a competition.

    Args:
        analysis: Structured competition analysis from competition_analysis.py.
        client: LLM client.
        model: Model name.
        max_num_approaches: Maximum number of approaches to generate.
        num_reflections: Number of refinement rounds per approach.
        seed_approaches: Optional pre-defined seed approaches.

    Returns:
        List of approach dictionaries.
    """
    approach_archive = []

    # Load seed approaches if provided
    if seed_approaches:
        for seed in seed_approaches:
            approach_archive.append(seed)
        print(f"Loaded {len(seed_approaches)} seed approaches.")

    # Prepare analysis string
    analysis_str = json.dumps(
        {k: v for k, v in analysis.items() if k != "raw_data"},
        indent=2,
    )

    for i in range(max_num_approaches):
        print(f"\nGenerating approach {i + 1}/{max_num_approaches}")

        try:
            prev_approaches_str = "\n\n".join(
                json.dumps(a, indent=2) for a in approach_archive
            )

            msg_history = []
            text, msg_history = get_response_from_llm(
                APPROACH_FIRST_PROMPT.format(
                    analysis=analysis_str,
                    prev_approaches=prev_approaches_str or "None yet.",
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=APPROACH_SYSTEM_PROMPT,
                msg_history=msg_history,
            )

            json_output = extract_json_between_markers(text)
            if json_output is None:
                print(f"Failed to extract JSON for approach {i + 1}")
                continue

            # Refinement rounds
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"  Refining: round {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        APPROACH_REFLECTION_PROMPT.format(
                            current_round=j + 2,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=APPROACH_SYSTEM_PROMPT,
                        msg_history=msg_history,
                    )
                    refined = extract_json_between_markers(text)
                    if refined is not None:
                        json_output = refined
                    if "I am done" in text:
                        print(f"  Approach converged after {j + 2} rounds.")
                        break

            approach_archive.append(json_output)
            print(f"  Generated: {json_output.get('Name', 'unnamed')}")

        except Exception as e:
            print(f"Failed to generate approach {i + 1}: {e}")
            continue

    return approach_archive


def rank_approaches(
    approaches: List[Dict],
    analysis: Dict,
    client: Any,
    model: str,
) -> List[Dict]:
    """
    Rank approaches by expected effectiveness for the competition.

    Returns approaches sorted by expected score (best first).
    """
    if len(approaches) <= 1:
        return approaches

    ranking_prompt = f"""You are evaluating solution approaches for an ML competition.

## Competition Analysis
{json.dumps({k: v for k, v in analysis.items() if k != "raw_data"}, indent=2)}

## Approaches to Rank
{json.dumps(approaches, indent=2)}

Rank these approaches from most promising to least promising.
Consider: expected score, feasibility, implementation complexity, and risk.

Respond with a JSON list of approach names in order from best to worst:

```json
{{"ranking": ["approach_name_1", "approach_name_2", ...]}}
```
"""

    text, _ = get_response_from_llm(
        ranking_prompt,
        client=client,
        model=model,
        system_message=APPROACH_SYSTEM_PROMPT,
    )

    ranking = extract_json_between_markers(text)
    if ranking and "ranking" in ranking:
        name_to_approach = {a.get("Name", ""): a for a in approaches}
        ranked = []
        for name in ranking["ranking"]:
            if name in name_to_approach:
                ranked.append(name_to_approach[name])
        # Add any approaches not in the ranking at the end
        for a in approaches:
            if a not in ranked:
                ranked.append(a)
        return ranked

    return approaches


def select_best_approach(
    approaches: List[Dict],
    analysis: Dict,
    client: Any,
    model: str,
) -> Dict:
    """Select the single best approach from ranked list."""
    ranked = rank_approaches(approaches, analysis, client, model)
    return ranked[0] if ranked else approaches[0]
