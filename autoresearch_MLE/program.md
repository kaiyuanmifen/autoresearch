# autoresearch_MLE

This is the autoresearch experimentation loop adapted to run as an autonomous
agent on **MLE-Bench** (OpenAI's machine-learning-engineering benchmark of
real Kaggle competitions).

The original `autoresearch` agent (see `../autoresearch/program.md`) is wired
to a single LLM-training task with a fixed 5-minute time budget. The MLE
adaptation keeps the same loop shape but swaps the inner task: instead of
minimising `val_bpb` on a tiny language model, the agent now iterates on a
solution to a Kaggle competition and minimises whatever metric the
competition uses (lower-or-higher-is-better is determined per task).

## Setup

The agent runs *inside* an MLE-Bench task container. By convention MLE-Bench
mounts these directories into the container:

| Path                 | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `/home/data/`        | Read-only competition data + `description.md`            |
| `/home/submission/`  | The agent writes `submission.csv` here for grading       |
| `/home/code/`        | The agent's working directory for any code it creates    |
| `/home/logs/`        | Anything the agent wants persisted as a log              |

To set up a new run, work with the user to:

1. **Pick a task**: agree on which MLE-Bench competition to attack. The list
   lives in `../mle-bench/experiments/splits/`. Use a short tag, e.g.
   `spaceship-titanic` or `aerial-cactus`.
2. **Verify mounts**: `/home/data/description.md`,
   `/home/data/sample_submission.csv` (if any), and the data files referenced
   in the description must all exist. If they don't, tell the user to launch
   the task with `mlebench run-agent ...`.
3. **Read the in-scope files**:
   - `/home/data/description.md` — the competition prompt. Read it carefully:
     it tells you the task, the data layout, the evaluation metric, the
     submission format, and any rules.
   - `sample_submission.csv` if present — defines the exact submission
     schema (column names, row order, dtypes).
   - A small head of each data file (`head -n 5`) so you understand the
     schema before writing code. **Do not** load entire datasets into your
     context — they can be huge.
4. **Decide where solutions live**: create `/home/code/solution.py` as the
   single file you will edit. Treat it the way `train.py` is treated in the
   original autoresearch loop: it is the only file you modify.
5. **Initialise `results.tsv`**: create `/home/logs/results.tsv` with the
   header row. The first run records the baseline.
6. **Confirm and go**.

## What you CAN do

- Modify `/home/code/solution.py` freely. Architecture, features, model
  family, hyperparameters, ensembling, cross-validation strategy — all fair
  game.
- Use any package already installed in the container image. The base image
  is built from `Dockerfile` in this directory.
- Read the data, write intermediate artifacts under `/home/code/artifacts/`,
  and cache anything expensive there.
- Use `git` *inside* `/home/code/` to checkpoint experiments — the loop
  relies on git for advance/rewind semantics, the same as the original
  autoresearch loop.

## What you CANNOT do

- **Touch `/home/data/`.** It is read-only and any write attempt will be
  rejected by the container.
- **Look at the test labels or the held-out portion of the data** beyond
  what `description.md` exposes — that is leaderboard cheating and grading
  will detect it.
- **Install new system packages** during the run. If something is missing,
  log it and work around it; the container is sealed.
- **Call out to the network** for anything other than the OpenAI/Anthropic
  API endpoint that the harness routes through. No `pip install`, no
  `wget`-ing extra datasets, no scraping. MLE-Bench enforces this.
- **Modify the grading harness.** `mlebench grade-sample` is the ground
  truth.

## The metric

Read `description.md`. Every competition specifies its own metric and the
direction of improvement. Common cases:

- AUC, accuracy, F1, mAP — **higher is better**.
- RMSE, log-loss, MAE, MAPE — **lower is better**.

When you log to `results.tsv`, always store the *raw* metric value as the
competition reports it, plus a `direction` column (`min` or `max`) so that
the loop logic knows which way is "better". Do not silently negate scores.

## Running an experiment

The inner loop is: edit `solution.py`, run it, capture the score, decide
keep/discard. The runner is:

```bash
uv run python /home/code/solution.py \
    --data-dir /home/data \
    --out /home/submission/submission.csv \
    > /home/logs/run.log 2>&1
```

This convention is enforced by `agent/loop.py` — `solution.py` MUST accept
`--data-dir` and `--out` and write a valid submission CSV to `--out`.

After the run, the loop calls a *local validation* hook
(`agent/grading.py::local_score`) that re-creates the same metric on a
held-out fold the agent has carved out from the training data. **Never**
score against `/home/data/test.csv` directly — there are no labels there.

The wall-clock budget per inner experiment defaults to **15 minutes**
(configurable via `MLE_EXP_BUDGET_SEC`). The total wall-clock budget for
the whole run is whatever MLE-Bench gives the container, typically 24h.

## Output format

`solution.py` must print, on its last line of stdout, a JSON blob like:

```
{"local_score": 0.8413, "direction": "max", "n_train": 9000, "n_val": 1000, "wall_seconds": 482.1}
```

The loop greps for `^{"local_score":` to extract the score. If the script
crashes, this line is missing and the run is logged as `crash`.

## Logging results

`/home/logs/results.tsv` (tab-separated) with these columns:

```
commit	local_score	direction	wall_seconds	status	description
```

1. `commit`: short git hash from the in-container `/home/code` git repo
2. `local_score`: the validation score (always the raw metric value)
3. `direction`: `min` or `max` — copied from the JSON blob, never inferred
4. `wall_seconds`: wall-clock time of the run (excluding harness overhead)
5. `status`: `keep`, `discard`, or `crash`
6. `description`: short text — what this experiment tried

**Do not commit `results.tsv`** in the agent's git repo — it should be in
`.gitignore`. It lives only in `/home/logs/`.

## The experiment loop

LOOP FOREVER (until MLE-Bench kills the container):

1. Inspect git state in `/home/code/`.
2. Edit `solution.py` with one experimental idea. **One idea per loop
   iteration.** Don't combine three changes into one commit — you can't
   tell which one helped.
3. `git commit -am "<short description of the idea>"`.
4. Run the experiment via `agent/loop.py run` (which handles timeout,
   logging, score extraction, and writing the submission to
   `/home/submission/submission.csv`).
5. If the run produced a valid submission **and** local_score improved in
   the right direction, advance the branch (keep the commit) and copy the
   submission to `/home/submission/submission.csv` so MLE-Bench picks it up.
6. If local_score is equal or worse, `git reset --hard HEAD~1` and move on.
7. If the run crashed, decide: dumb bug → fix and retry once; fundamental
   → log as `crash`, reset, move on.
8. Append a row to `/home/logs/results.tsv`.
9. Go to 1.

**Always keep `/home/submission/submission.csv` valid.** Even if your
current iteration crashes, the previous best submission must remain in
place — MLE-Bench grades whatever is at that path when the container's
time runs out. The loop helper enforces this by only overwriting the
submission file when the new score is an improvement.

**NEVER STOP.** Same rule as the original autoresearch loop. The container
will be killed by MLE-Bench when its budget expires; until then, keep
iterating. Don't ask the human if you should continue — they're not
watching.

## Differences from the original autoresearch loop

- The metric is task-dependent and the *direction* of improvement is
  task-dependent. The original loop hard-coded "lower val_bpb is better".
- The artifact that gets graded is `submission.csv`, not the model.
  Improvements that help local validation but break the submission file
  format are worthless.
- Per-experiment budget is 15 minutes (vs. 5), because Kaggle-style models
  often need longer to converge meaningfully.
- The agent must protect `submission.csv` — it is the single source of
  truth at grading time.
- The container is sealed: no network, no `pip install`, no escape hatches.
