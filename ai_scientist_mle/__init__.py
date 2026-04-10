"""
AI Scientist for MLE-Bench: An autonomous ML engineering agent.

Adapts the Sakana AI "AI Scientist" architecture for solving
MLE-bench (Kaggle-style) machine learning competitions.

Pipeline:
    1. Competition Analysis - Parse problem, data, and metrics
    2. Approach Generation - Generate candidate solution strategies
    3. Solution Implementation - Write ML code via LLM-driven coding
    4. Experiment Execution - Run solutions and capture results
    5. Solution Review - Evaluate results and identify improvements
    6. Iteration - Refine based on review feedback
    7. Submission - Produce final prediction CSV
"""

__version__ = "0.1.0"
