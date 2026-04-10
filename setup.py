from setuptools import setup, find_packages

setup(
    name="ai-scientist-mle",
    version="0.1.0",
    description="AI Scientist for MLE-Bench: Autonomous ML Competition Solver",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="AutoResearch",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        "backoff>=2.2.1",
        "aider-chat>=0.40.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "tensorflow": ["tensorflow>=2.13.0"],
        "automl": ["flaml>=2.0.0"],
        "all": [
            "torch>=2.0.0",
            "tensorflow>=2.13.0",
            "catboost>=1.2.0",
            "flaml>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mle-scientist=launch_mle_scientist:main",
        ],
    },
)
