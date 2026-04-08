#!/usr/bin/env bash
# Entrypoint for the autoresearch_MLE container.
#
# Layout (provided by MLE-Bench):
#   /home/data        read-only competition data + description.md
#   /home/submission  agent writes submission.csv here
#   /home/code        agent's working directory (we git-init it)
#   /home/logs        agent's log directory
#
# This script:
#   1. Initialises /home/code as a git repo and writes a stub solution.py.
#   2. Hands off to the LLM agent driver (the actual experimentation is
#      done by an LLM that calls `python -m agent.loop run` repeatedly).
#   3. If no LLM driver is wired up, falls back to running the stub
#      solution once so the container always produces a valid (if poor)
#      submission.

set -euo pipefail

export MLE_DATA_DIR="${MLE_DATA_DIR:-/home/data}"
export MLE_SUBMISSION_DIR="${MLE_SUBMISSION_DIR:-/home/submission}"
export MLE_CODE_DIR="${MLE_CODE_DIR:-/home/code}"
export MLE_LOGS_DIR="${MLE_LOGS_DIR:-/home/logs}"
export MLE_EXP_BUDGET_SEC="${MLE_EXP_BUDGET_SEC:-900}"

mkdir -p "$MLE_CODE_DIR" "$MLE_SUBMISSION_DIR" "$MLE_LOGS_DIR"

cd /opt/autoresearch_mle

# 1. Initialise the working tree.
python -m agent.loop init

# 2. Always seed a baseline submission so the grader has *something* to
#    grade, even if every subsequent iteration crashes. The stub solution
#    just copies sample_submission.csv if one exists.
python -m agent.loop run --description "baseline (stub)" || true

# 3. Hand off to the LLM driver if one is configured. The driver is
#    expected to call `python -m agent.loop run` repeatedly while editing
#    /home/code/solution.py between iterations. By convention the driver
#    binary is /opt/autoresearch_mle/driver, but we tolerate it being
#    absent so the image is still useful for debugging.
if [[ -x /opt/autoresearch_mle/driver ]]; then
    exec /opt/autoresearch_mle/driver
fi

if [[ -n "${MLE_AGENT_DRIVER:-}" ]]; then
    exec ${MLE_AGENT_DRIVER}
fi

echo "no LLM driver configured (set MLE_AGENT_DRIVER or install /opt/autoresearch_mle/driver)" >&2
echo "the stub baseline submission has been written to $MLE_SUBMISSION_DIR/submission.csv" >&2
echo "sleeping for the remainder of the MLE-Bench budget so the container stays alive" >&2

# Keep the container alive so MLE-Bench can grade the stub. The harness
# will SIGKILL us when its budget expires.
exec sleep infinity
