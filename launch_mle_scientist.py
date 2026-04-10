#!/usr/bin/env python3
"""
AI Scientist for MLE-Bench - Main Entry Point

Adapts the Sakana AI "AI Scientist" autonomous research pipeline
to solve MLE-bench (Kaggle-style) machine learning competitions.

Pipeline:
    1. Competition Analysis  - Parse problem, data, metrics
    2. Approach Generation   - Generate candidate solution strategies
    3. Solution Implementation - LLM-driven coding (via aider)
    4. Experiment Execution  - Run solutions, capture results
    5. Solution Review       - Evaluate and suggest improvements
    6. Iteration             - Refine based on feedback
    7. Submission            - Produce final prediction CSV

Usage:
    python launch_mle_scientist.py \
        --competition-dir /path/to/competition/data \
        --model claude-sonnet-4-6 \
        --num-approaches 3

    # Run on all competitions in a directory:
    python launch_mle_scientist.py \
        --competitions-dir /path/to/all/competitions \
        --model gpt-4o \
        --parallel 4
"""

import argparse
import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist_mle.llm import create_client, AVAILABLE_LLMS
from ai_scientist_mle.competition_analysis import analyze_competition, inspect_data_files
from ai_scientist_mle.generate_approaches import (
    generate_approaches,
    rank_approaches,
    select_best_approach,
)
from ai_scientist_mle.perform_experiments import (
    create_solution_workspace,
    perform_experiments,
)
from ai_scientist_mle.perform_review import (
    perform_review,
    perform_improvement,
    should_continue_iterating,
)
from ai_scientist_mle.submission import (
    find_best_submission,
    validate_submission,
    prepare_mle_bench_submission,
    collect_all_submissions,
    generate_submission_report,
)
from ai_scientist_mle.perform_writeup import (
    generate_writeup,
    generate_summary_report,
)

NUM_REFLECTIONS = 3
MAX_REVIEW_ITERATIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Scientist for MLE-Bench: Autonomous ML Competition Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single competition:
  python launch_mle_scientist.py \\
      --competition-dir ./data/titanic \\
      --model claude-sonnet-4-6

  # All competitions in a directory:
  python launch_mle_scientist.py \\
      --competitions-dir ./data/competitions \\
      --model gpt-4o --parallel 4

  # With custom settings:
  python launch_mle_scientist.py \\
      --competition-dir ./data/titanic \\
      --model gpt-4.1 \\
      --num-approaches 5 \\
      --max-experiment-runs 8 \\
      --timeout 7200
        """,
    )

    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--competition-dir",
        type=str,
        help="Path to a single competition data directory.",
    )
    input_group.add_argument(
        "--competitions-dir",
        type=str,
        help="Path to directory containing multiple competition subdirectories.",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        choices=AVAILABLE_LLMS,
        help="LLM model to use (default: claude-sonnet-4-6).",
    )

    # Pipeline control
    parser.add_argument(
        "--num-approaches",
        type=int,
        default=3,
        help="Number of solution approaches to generate (default: 3).",
    )
    parser.add_argument(
        "--max-experiment-runs",
        type=int,
        default=5,
        help="Maximum experiment runs per approach (default: 5).",
    )
    parser.add_argument(
        "--max-review-iterations",
        type=int,
        default=3,
        help="Maximum review-improve iterations (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per experiment run in seconds (default: 3600).",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=3,
        help="Number of LLM reflection rounds (default: 3).",
    )

    # Skip options
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip competition analysis (load existing).",
    )
    parser.add_argument(
        "--skip-approach-generation",
        action="store_true",
        help="Skip approach generation (load existing).",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="Skip report generation.",
    )

    # Execution configuration
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel competitions to process (default: 0 = sequential).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store results (default: results).",
    )
    parser.add_argument(
        "--seed-approaches",
        type=str,
        default=None,
        help="Path to JSON file with seed approaches.",
    )

    return parser.parse_args()


def setup_coder(model: str, workspace_dir: str, idea_name: str = "solution"):
    """Create an aider Coder instance for the workspace."""
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model

    solution_file = osp.join(workspace_dir, "solution.py")
    notes_file = osp.join(workspace_dir, "notes.txt")
    fnames = [solution_file, notes_file]

    io = InputOutput(
        yes=True,
        chat_history_file=osp.join(workspace_dir, f"{idea_name}_aider.txt"),
    )

    if model == "deepseek-coder-v2-0724":
        main_model = Model("deepseek/deepseek-coder")
    elif model == "deepseek-reasoner":
        main_model = Model("deepseek/deepseek-reasoner")
    elif model == "llama3.1-405b":
        main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    else:
        main_model = Model(model)

    coder = Coder.create(
        main_model=main_model,
        fnames=fnames,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",
    )

    return coder


def do_competition(
    competition_dir: str,
    results_dir: str,
    model: str,
    client: Any,
    client_model: str,
    num_approaches: int = 3,
    max_experiment_runs: int = 5,
    max_review_iterations: int = 3,
    timeout: int = 3600,
    num_reflections: int = 3,
    skip_analysis: bool = False,
    skip_approach_generation: bool = False,
    skip_writeup: bool = False,
    seed_approaches: Optional[List[Dict]] = None,
    log_file: bool = False,
) -> bool:
    """
    Run the full AI Scientist MLE pipeline on a single competition.

    Returns True if the competition was solved successfully.
    """
    competition_id = osp.basename(osp.normpath(competition_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = osp.join(results_dir, f"{timestamp}_{competition_id}")
    os.makedirs(workspace_dir, exist_ok=True)

    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(workspace_dir, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log

    try:
        print_time()
        print(f"{'='*60}")
        print(f"Starting AI Scientist MLE for: {competition_id}")
        print(f"{'='*60}")

        # ============================================================
        # PHASE 1: COMPETITION ANALYSIS
        # ============================================================
        print_time()
        print("\n[Phase 1] Analyzing competition...")

        analysis_path = osp.join(workspace_dir, "analysis.json")
        if skip_analysis and osp.exists(analysis_path):
            with open(analysis_path, "r") as f:
                analysis = json.load(f)
            print("Loaded existing analysis.")
        else:
            analysis = analyze_competition(
                competition_dir, client, client_model, competition_id
            )
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis complete: {analysis.get('problem_type', 'unknown')} problem")

        # Inspect data files
        data_summary = inspect_data_files(competition_dir, client, client_model)
        data_summary_path = osp.join(workspace_dir, "data_summary.json")
        with open(data_summary_path, "w") as f:
            json.dump(data_summary, f, indent=2, default=str)

        # ============================================================
        # PHASE 2: APPROACH GENERATION
        # ============================================================
        print_time()
        print("\n[Phase 2] Generating solution approaches...")

        approaches_path = osp.join(workspace_dir, "approaches.json")
        if skip_approach_generation and osp.exists(approaches_path):
            with open(approaches_path, "r") as f:
                approaches = json.load(f)
            print(f"Loaded {len(approaches)} existing approaches.")
        else:
            approaches = generate_approaches(
                analysis,
                client=client,
                model=client_model,
                max_num_approaches=num_approaches,
                num_reflections=num_reflections,
                seed_approaches=seed_approaches,
            )
            with open(approaches_path, "w") as f:
                json.dump(approaches, f, indent=2)
            print(f"Generated {len(approaches)} approaches.")

        # Rank and select best approach
        approaches = rank_approaches(approaches, analysis, client, client_model)
        best_approach = approaches[0]
        print(f"Selected approach: {best_approach.get('Name', 'unknown')}")

        # ============================================================
        # PHASE 3: SOLUTION IMPLEMENTATION & EXPERIMENTS
        # ============================================================
        print_time()
        print("\n[Phase 3] Implementing solution and running experiments...")

        create_solution_workspace(best_approach, analysis, workspace_dir)

        # Set up the aider coder
        coder = setup_coder(model, workspace_dir, competition_id)

        success, run_results = perform_experiments(
            approach=best_approach,
            analysis=analysis,
            workspace_dir=workspace_dir,
            competition_dir=competition_dir,
            coder=coder,
            timeout=timeout,
        )

        # Save run results
        results_path = osp.join(workspace_dir, "run_results.json")
        with open(results_path, "w") as f:
            json.dump(run_results, f, indent=2)

        if not success:
            print("WARNING: Experiments did not complete successfully.")

        # ============================================================
        # PHASE 4: REVIEW & ITERATION
        # ============================================================
        print_time()
        print("\n[Phase 4] Reviewing solution...")

        review = None
        for iteration in range(max_review_iterations):
            print(f"\n  Review iteration {iteration + 1}/{max_review_iterations}")

            review = perform_review(
                workspace_dir=workspace_dir,
                analysis=analysis,
                approach=best_approach,
                client=client,
                model=client_model,
                num_reflections=num_reflections,
            )

            print(f"  Review score: {review.get('Overall_Score', 'N/A')}/10")
            print(f"  Decision: {review.get('Decision', 'N/A')}")

            if not should_continue_iterating(review, iteration, max_review_iterations):
                print("  Review passed. Moving to submission.")
                break

            # Apply improvements
            print("  Applying improvements...")
            perform_improvement(review, coder)

            # Re-run experiments after improvement
            print("  Re-running experiments...")
            _, new_results = perform_experiments(
                approach=best_approach,
                analysis=analysis,
                workspace_dir=workspace_dir,
                competition_dir=competition_dir,
                coder=coder,
                timeout=timeout,
            )
            run_results.extend(new_results)

        # ============================================================
        # PHASE 5: SUBMISSION
        # ============================================================
        print_time()
        print("\n[Phase 5] Preparing submission...")

        best_submission = find_best_submission(workspace_dir)
        if best_submission:
            validation = validate_submission(
                best_submission,
                sample_submission_path=osp.join(competition_dir, "sample_submission.csv"),
            )
            print(f"Submission validation: {'PASS' if validation['valid'] else 'FAIL'}")
            if validation["errors"]:
                print(f"  Errors: {validation['errors']}")
            if validation["warnings"]:
                print(f"  Warnings: {validation['warnings']}")

            # Prepare MLE-bench format
            submission_dir = osp.join(workspace_dir, "final_submission")
            prepare_mle_bench_submission(
                best_submission, competition_id, submission_dir,
            )
            print(f"Submission prepared in {submission_dir}")
        else:
            print("WARNING: No submission generated!")

        # ============================================================
        # PHASE 6: WRITEUP
        # ============================================================
        if not skip_writeup:
            print_time()
            print("\n[Phase 6] Generating report...")

            generate_writeup(
                workspace_dir=workspace_dir,
                analysis=analysis,
                approach=best_approach,
                results=run_results,
                review=review,
                client=client,
                model=client_model,
            )

        # ============================================================
        # DONE
        # ============================================================
        print_time()
        print(f"\n{'='*60}")
        print(f"Completed: {competition_id}")
        print(f"Workspace: {workspace_dir}")
        print(f"Success: {success}")
        print(f"Submission: {'Generated' if best_submission else 'MISSING'}")
        print(f"{'='*60}")

        return success

    except Exception as e:
        print(f"Failed to process competition {competition_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


def worker(
    queue: multiprocessing.Queue,
    results_dir: str,
    model: str,
    client: Any,
    client_model: str,
    args: argparse.Namespace,
    gpu_id: int,
):
    """Worker process for parallel competition processing."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")

    while True:
        item = queue.get()
        if item is None:
            break

        competition_dir = item
        competition_id = osp.basename(competition_dir)
        print(f"Worker {gpu_id}: Processing {competition_id}")

        success = do_competition(
            competition_dir=competition_dir,
            results_dir=results_dir,
            model=model,
            client=client,
            client_model=client_model,
            num_approaches=args.num_approaches,
            max_experiment_runs=args.max_experiment_runs,
            max_review_iterations=args.max_review_iterations,
            timeout=args.timeout,
            num_reflections=args.num_reflections,
            skip_analysis=args.skip_analysis,
            skip_approach_generation=args.skip_approach_generation,
            skip_writeup=args.skip_writeup,
            log_file=True,
        )

        print(f"Worker {gpu_id}: Completed {competition_id}, Success: {success}")

    print(f"Worker {gpu_id} finished.")


def main():
    args = parse_arguments()

    # Create client
    client, client_model = create_client(args.model)

    # Set up results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Load seed approaches if provided
    seed_approaches = None
    if args.seed_approaches:
        with open(args.seed_approaches, "r") as f:
            seed_approaches = json.load(f)

    # Determine competitions to process
    if args.competition_dir:
        # Single competition
        competitions = [args.competition_dir]
    else:
        # Multiple competitions
        competitions = []
        for d in sorted(os.listdir(args.competitions_dir)):
            comp_path = osp.join(args.competitions_dir, d)
            if osp.isdir(comp_path):
                competitions.append(comp_path)
        print(f"Found {len(competitions)} competitions to process.")

    if not competitions:
        print("No competitions found. Exiting.")
        sys.exit(1)

    # Process competitions
    if args.parallel > 0 and len(competitions) > 1:
        # Parallel processing
        print(f"Running {args.parallel} parallel workers")

        try:
            import torch
            available_gpus = list(range(torch.cuda.device_count()))
        except ImportError:
            available_gpus = [0]

        if args.parallel > len(available_gpus):
            print(
                f"Warning: Requested {args.parallel} workers, "
                f"but only {len(available_gpus)} GPUs available."
            )
            args.parallel = max(len(available_gpus), 1)

        queue = multiprocessing.Queue()
        for comp in competitions:
            queue.put(comp)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else 0
            p = multiprocessing.Process(
                target=worker,
                args=(queue, args.results_dir, args.model, client, client_model, args, gpu_id),
            )
            p.start()
            time.sleep(5)  # Stagger process starts
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel workers completed.")
    else:
        # Sequential processing
        results = {}
        for comp_dir in competitions:
            comp_id = osp.basename(comp_dir)
            print(f"\nProcessing competition: {comp_id}")
            success = do_competition(
                competition_dir=comp_dir,
                results_dir=args.results_dir,
                model=args.model,
                client=client,
                client_model=client_model,
                num_approaches=args.num_approaches,
                max_experiment_runs=args.max_experiment_runs,
                max_review_iterations=args.max_review_iterations,
                timeout=args.timeout,
                num_reflections=args.num_reflections,
                skip_analysis=args.skip_analysis,
                skip_approach_generation=args.skip_approach_generation,
                skip_writeup=args.skip_writeup,
                seed_approaches=seed_approaches,
            )
            results[comp_id] = {"success": success}

    # Generate final submission collection and report
    print("\n" + "=" * 60)
    print("Collecting all submissions...")
    submission_dir = osp.join(args.results_dir, "final_submissions")
    collect_all_submissions(args.results_dir, submission_dir)

    report = generate_submission_report(args.results_dir)
    print(f"\nFinal Report:")
    print(f"  Total competitions: {report['total_competitions']}")
    print(f"  Submissions generated: {report['submissions_generated']}")
    print(f"  Submissions missing: {report['submissions_missing']}")

    # Save report
    report_path = osp.join(args.results_dir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {args.results_dir}")
    print("All competitions processed.")


if __name__ == "__main__":
    main()
