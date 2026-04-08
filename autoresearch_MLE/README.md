# autoresearch_MLE

Adaptation of the [`autoresearch`](../autoresearch) experimentation loop to
[OpenAI's MLE-Bench](../mle-bench): instead of minimising val_bpb on a tiny
language model, the agent iterates on a Kaggle competition solution and
optimises whatever metric the competition specifies.

## What's here

```
autoresearch_MLE/
├── program.md         # the agent's specification (read this first)
├── Dockerfile         # MLE-Bench-compatible container image
├── entrypoint.sh      # container entrypoint
├── pyproject.toml     # Python deps
├── agent/
│   ├── loop.py        # inner experiment loop (init / run / best subcommands)
│   ├── submission.py  # submission helpers (atomic write, schema check)
│   └── grading.py     # local validation metrics (rmse, auc, log_loss, ...)
└── tests/
    └── test_loop_helpers.py
```

## How it differs from the original autoresearch loop

| Aspect              | autoresearch (LLM training)        | autoresearch_MLE (MLE-Bench)               |
|---------------------|------------------------------------|--------------------------------------------|
| Inner script        | `train.py`                         | `/home/code/solution.py`                   |
| Metric              | val_bpb (lower is better, fixed)   | task-dependent, *direction* per-task       |
| Per-iteration budget| 5 min                              | 15 min (env: `MLE_EXP_BUDGET_SEC`)         |
| Total budget        | wall-clock until human stops       | whatever MLE-Bench grants the container    |
| Graded artifact     | the val_bpb log line               | `/home/submission/submission.csv`          |
| Sealed env          | no                                 | yes (no network, no `pip install`)         |

The loop logic — propose, commit, run, advance-or-rewind — is the same.

## Quickstart (outside MLE-Bench, for development)

```bash
# Set up a fake task in /tmp
mkdir -p /tmp/mle/{data,code,logs,submission}
echo "id,label"$'\n'"1,0" > /tmp/mle/data/sample_submission.csv

export MLE_DATA_DIR=/tmp/mle/data
export MLE_CODE_DIR=/tmp/mle/code
export MLE_LOGS_DIR=/tmp/mle/logs
export MLE_SUBMISSION_DIR=/tmp/mle/submission
export PYTHONPATH=$PWD

python -m agent.loop init   # writes solution.py stub + git repo + results.tsv
python -m agent.loop run    # one iteration; writes submission.csv if improved
python -m agent.loop best   # print best kept score so far
```

## Running inside MLE-Bench

Build the image and register it as an agent:

```bash
docker build -t autoresearch-mle:latest autoresearch_MLE/
mlebench run-agent \
    --agent-image autoresearch-mle:latest \
    --task spaceship-titanic \
    ...
```

The container's entrypoint (`entrypoint.sh`):

1. Initialises `/home/code` as a git repo and writes a stub `solution.py`.
2. Runs the stub once so `submission.csv` always exists for the grader.
3. Hands off to the LLM driver (configured via `$MLE_AGENT_DRIVER` or
   `/opt/autoresearch_mle/driver`), which calls
   `python -m agent.loop run` repeatedly while editing `solution.py` between
   iterations.
4. If no driver is configured, sleeps until the harness kills the container
   (so MLE-Bench still grades the stub baseline rather than getting nothing).

## Tests

```bash
PYTHONPATH=. python -m pytest tests/
```

The tests cover the pure-function helpers (score-blob extraction,
better/worse comparison, all metric implementations). The end-to-end loop
behaviour is covered by the smoke procedure documented in `program.md`.

## Status

This is the initial scaffold. The pieces in place are:

- [x] `program.md` — the agent's specification
- [x] `agent/loop.py` — keep/discard loop, score extraction, advance/rewind
- [x] `agent/submission.py` — atomic submission writes + schema check
- [x] `agent/grading.py` — common local validation metrics
- [x] `Dockerfile` + `entrypoint.sh` — MLE-Bench-compatible container
- [x] Unit tests for helpers
- [ ] LLM driver (the thing that actually edits `solution.py` between
      iterations) — pluggable via `$MLE_AGENT_DRIVER`; not shipped here
- [ ] Per-task baseline solutions for the MLE-Bench dev split
- [ ] Integration with MLE-Bench's grading harness
