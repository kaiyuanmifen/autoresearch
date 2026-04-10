# AI Scientist for MLE-Bench

An autonomous ML engineering agent that solves [MLE-bench](https://github.com/openai/mle-bench) competitions using the [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) architecture.

## Overview

This project adapts the AI Scientist's autonomous research pipeline for Kaggle-style ML competitions. Instead of generating research papers, it generates competition solutions:

| AI Scientist Phase | MLE-Bench Adaptation |
|---|---|
| Idea Generation | Approach Generation (solution strategies) |
| Experiment Execution | Solution Implementation & Training |
| Paper Writing | Technical Report Generation |
| Peer Review | Solution Review & Iteration |

## Pipeline

```
┌─────────────────────┐
│ Competition Analysis │ ← Parse problem, data, metrics
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Approach Generation  │ ← Generate N candidate strategies
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Rank & Select Best   │ ← LLM-based approach ranking
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Solution Development │ ← LLM writes code via aider
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Experiment Execution │ ← Run solution, capture results
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Solution Review      │ ← Evaluate quality, suggest fixes
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Iterate (if needed)  │ ← Apply improvements, re-run
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Generate Submission  │ ← Produce submission.csv + JSONL
└─────────────────────┘
```

## Installation

```bash
pip install -e .

# Or with all optional ML frameworks:
pip install -e ".[all]"
```

## Configuration

Set your API key as an environment variable:

```bash
# For Anthropic models (recommended)
export ANTHROPIC_API_KEY="your-key"

# For OpenAI models
export OPENAI_API_KEY="your-key"

# For DeepSeek models
export DEEPSEEK_API_KEY="your-key"

# For Gemini models
export GEMINI_API_KEY="your-key"
```

## Usage

### Single Competition

```bash
python launch_mle_scientist.py \
    --competition-dir /path/to/competition/data \
    --model claude-sonnet-4-6 \
    --num-approaches 3
```

### Multiple Competitions

```bash
python launch_mle_scientist.py \
    --competitions-dir /path/to/all/competitions \
    --model gpt-4o \
    --parallel 4
```

### With MLE-Bench

```bash
# 1. Prepare MLE-bench data
mlebench prepare --lite

# 2. Run AI Scientist MLE on all competitions
python launch_mle_scientist.py \
    --competitions-dir ~/.mlebench/data \
    --model claude-sonnet-4-6 \
    --results-dir ./results

# 3. Grade submissions
mlebench grade --submission results/final_submissions/submissions.jsonl
```

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--competition-dir` | - | Path to single competition |
| `--competitions-dir` | - | Path to multiple competitions |
| `--model` | `claude-sonnet-4-6` | LLM model to use |
| `--num-approaches` | 3 | Number of approaches to generate |
| `--max-experiment-runs` | 5 | Max experiment runs per approach |
| `--max-review-iterations` | 3 | Max review-improve cycles |
| `--timeout` | 3600 | Timeout per run (seconds) |
| `--parallel` | 0 | Number of parallel workers |
| `--results-dir` | `results` | Output directory |
| `--skip-analysis` | false | Skip competition analysis |
| `--skip-approach-generation` | false | Skip approach generation |
| `--skip-writeup` | false | Skip report generation |
| `--seed-approaches` | - | Path to seed approaches JSON |

## Project Structure

```
autoresearch/
├── ai_scientist_mle/           # Core library
│   ├── __init__.py
│   ├── llm.py                  # LLM client abstraction
│   ├── competition_analysis.py # Competition parsing & analysis
│   ├── generate_approaches.py  # Solution strategy generation
│   ├── perform_experiments.py  # Solution execution & testing
│   ├── perform_review.py       # Solution quality review
│   ├── perform_writeup.py      # Report generation
│   └── submission.py           # Submission handling
├── templates/
│   └── mle_bench/
│       ├── prompt.json         # System prompts
│       └── seed_approaches.json # Pre-defined seed approaches
├── launch_mle_scientist.py     # Main entry point
├── requirements.txt
└── setup.py
```

## Supported Models

- **Anthropic**: Claude Sonnet 4.6, Claude Opus 4.6, Claude 3.5 Sonnet
- **OpenAI**: GPT-4o, GPT-4.1, o1, o3-mini
- **DeepSeek**: DeepSeek Chat, DeepSeek Coder, DeepSeek Reasoner
- **Google**: Gemini 2.0 Flash, Gemini 2.5 Pro
- **Meta** (via OpenRouter): LLaMA 3.1 405B

## How It Works

### 1. Competition Analysis
The LLM analyzes the competition description, data files, evaluation metric, and submission format to produce a structured understanding of the problem.

### 2. Approach Generation
Multiple solution strategies are generated, each specifying a model, preprocessing pipeline, feature engineering steps, and implementation plan. Approaches are ranked by expected effectiveness.

### 3. Solution Implementation
An aider-powered coding agent implements the top-ranked approach in `solution.py`, modifying code iteratively based on experiment results.

### 4. Experiment Execution
Solutions are executed with `python solution.py --data_dir=<path> --out_dir=run_<i>`. Results are captured and fed back to the coding agent for refinement.

### 5. Solution Review
An LLM reviewer evaluates the solution's quality, identifies bugs, and suggests improvements. The review-improve loop continues until the solution meets quality thresholds.

### 6. Submission
The best submission.csv is validated and packaged in MLE-bench JSONL format for grading.

## Acknowledgments

- [Sakana AI - The AI Scientist](https://github.com/SakanaAI/AI-Scientist) for the autonomous research agent architecture
- [OpenAI - MLE-bench](https://github.com/openai/mle-bench) for the ML engineering benchmark
- [aider](https://github.com/paul-gauthier/aider) for the LLM-powered coding agent
