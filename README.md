# autoresearch on MLE-Bench

This is the meta-repo that wires the autonomous **autoresearch**
experimentation loop to **OpenAI's MLE-Bench** machine-learning-engineering
benchmark of real Kaggle competitions.

```
autoresearch/         # original autoresearch loop (LLM training, val_bpb)
                      #   gitlink → kaiyuanmifen/autoresearch
mle-bench/            # OpenAI's MLE-Bench harness + competitions
                      #   gitlink → openai/mle-bench
autoresearch_MLE/     # the bridge — autoresearch's loop, retargeted at
                      #   MLE-Bench tasks. This is what this branch builds.
```

The original `autoresearch/` loop (see its `program.md`) is a tight,
single-GPU, fixed-time-budget loop that has an LLM agent edit a
single Python file, run it, and decide keep/discard based on validation
loss. `autoresearch_MLE/` keeps the same loop shape but swaps the inner
task: the agent now edits `solution.py` for a Kaggle competition, scores
itself locally, and writes a submission CSV that MLE-Bench grades.

Read [`autoresearch_MLE/program.md`](autoresearch_MLE/program.md) for the
agent's specification, and [`autoresearch_MLE/README.md`](autoresearch_MLE/README.md)
for the developer quickstart.

## Status

Initial scaffold landed on the `MLE` branch:

- [x] `autoresearch_MLE/program.md` — agent spec adapted for MLE-Bench
- [x] `autoresearch_MLE/agent/loop.py` — keep/discard inner loop with
      score extraction, advance/rewind, submission protection
- [x] `autoresearch_MLE/agent/submission.py` — atomic submission writes
- [x] `autoresearch_MLE/agent/grading.py` — common local-validation metrics
- [x] `autoresearch_MLE/Dockerfile` + `entrypoint.sh` — MLE-Bench-compatible
      container
- [x] Unit tests for the helpers (18 passing)
- [ ] LLM driver that actually edits `solution.py` between iterations
      (pluggable via `$MLE_AGENT_DRIVER`)
- [ ] Wired-up integration with the MLE-Bench harness
- [ ] Per-task baseline solutions for the dev split
