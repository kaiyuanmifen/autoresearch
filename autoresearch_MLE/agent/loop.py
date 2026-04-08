"""Inner experiment loop for autoresearch_MLE.

This module is the runtime equivalent of ``train.py`` in the original
autoresearch project, but instead of training a single LLM it shells out to
``/home/code/solution.py`` and orchestrates the keep/discard decision around
each run. The actual experimental edits to ``solution.py`` are made by the
LLM agent driving the loop -- this file is the harness it relies on.

Usage::

    python -m agent.loop run                # one experiment iteration
    python -m agent.loop init               # set up /home/code + results.tsv
    python -m agent.loop best               # print the best score so far

Environment variables (all optional):

    MLE_DATA_DIR        default /home/data
    MLE_SUBMISSION_DIR  default /home/submission
    MLE_CODE_DIR        default /home/code
    MLE_LOGS_DIR        default /home/logs
    MLE_EXP_BUDGET_SEC  default 900   (per-iteration wall-clock budget)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("MLE_DATA_DIR", "/home/data"))
SUBMISSION_DIR = Path(os.environ.get("MLE_SUBMISSION_DIR", "/home/submission"))
CODE_DIR = Path(os.environ.get("MLE_CODE_DIR", "/home/code"))
LOGS_DIR = Path(os.environ.get("MLE_LOGS_DIR", "/home/logs"))

EXP_BUDGET_SEC = int(os.environ.get("MLE_EXP_BUDGET_SEC", "900"))

SOLUTION_PATH = CODE_DIR / "solution.py"
RESULTS_TSV = LOGS_DIR / "results.tsv"
RUN_LOG = LOGS_DIR / "run.log"
BEST_SUBMISSION = SUBMISSION_DIR / "submission.csv"
CANDIDATE_SUBMISSION = CODE_DIR / "candidate_submission.csv"

RESULTS_HEADER = "commit\tlocal_score\tdirection\twall_seconds\tstatus\tdescription"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    commit: str
    local_score: float | None
    direction: str | None  # "min" or "max"
    wall_seconds: float
    status: str  # keep / discard / crash
    description: str

    def to_tsv_row(self) -> str:
        score_repr = (
            f"{self.local_score:.6f}" if self.local_score is not None else "0.000000"
        )
        direction = self.direction or ""
        return "\t".join(
            [
                self.commit,
                score_repr,
                direction,
                f"{self.wall_seconds:.1f}",
                self.status,
                self.description.replace("\t", " ").replace("\n", " "),
            ]
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path = CODE_DIR, check: bool = True) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


def _short_sha() -> str:
    try:
        return _git("rev-parse", "--short=7", "HEAD")
    except subprocess.CalledProcessError:
        return "0000000"


def _is_better(new: float, old: float | None, direction: str) -> bool:
    if old is None:
        return True
    if direction == "max":
        return new > old
    if direction == "min":
        return new < old
    raise ValueError(f"unknown direction {direction!r}")


def _read_best_score() -> tuple[float | None, str | None]:
    """Return (best_score, direction) by scanning results.tsv for keep rows."""
    if not RESULTS_TSV.exists():
        return None, None
    best: float | None = None
    direction: str | None = None
    with RESULTS_TSV.open() as fh:
        next(fh, None)  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _, score_s, dir_s, _, status, *_ = parts
            if status != "keep":
                continue
            try:
                score = float(score_s)
            except ValueError:
                continue
            direction = direction or (dir_s or None)
            if direction is None:
                continue
            if _is_better(score, best, direction):
                best = score
    return best, direction


def _extract_score_blob(stdout: str) -> dict | None:
    """Find the last line of stdout that looks like the score JSON blob."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith('{"local_score"') and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def _ensure_dirs() -> None:
    for d in (CODE_DIR, LOGS_DIR, SUBMISSION_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    _ensure_dirs()
    if not (CODE_DIR / ".git").exists():
        _git("init", "-q")
        _git("config", "user.email", "autoresearch@local")
        _git("config", "user.name", "autoresearch")
        # The per-task repo is a throwaway scratchpad: it never leaves the
        # container and there is no signing key inside an MLE-Bench image.
        # Disable signing *locally* so commits succeed.
        _git("config", "commit.gpgsign", "false")
        _git("config", "tag.gpgsign", "false")
    if not SOLUTION_PATH.exists():
        SOLUTION_PATH.write_text(_DEFAULT_SOLUTION_TEMPLATE)
        _git("add", "solution.py")
        _git("commit", "-q", "-m", "initial solution stub")
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER + "\n")
    print(f"initialised: code={CODE_DIR} logs={LOGS_DIR} submission={SUBMISSION_DIR}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run a single experiment iteration around the current solution.py."""
    _ensure_dirs()
    if not SOLUTION_PATH.exists():
        print(f"error: {SOLUTION_PATH} does not exist; run `init` first", file=sys.stderr)
        return 2

    description = args.description or _git("log", "-1", "--pretty=%s")
    commit = _short_sha()

    if CANDIDATE_SUBMISSION.exists():
        CANDIDATE_SUBMISSION.unlink()

    cmd = [
        sys.executable,
        str(SOLUTION_PATH),
        "--data-dir",
        str(DATA_DIR),
        "--out",
        str(CANDIDATE_SUBMISSION),
    ]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=CODE_DIR,
            capture_output=True,
            text=True,
            timeout=EXP_BUDGET_SEC,
        )
        wall = time.monotonic() - t0
        stdout = proc.stdout
        stderr = proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        wall = time.monotonic() - t0
        stdout = exc.stdout.decode() if exc.stdout else ""
        stderr = (exc.stderr.decode() if exc.stderr else "") + "\n[timeout]"
        rc = -1

    RUN_LOG.write_text(
        f"$ {' '.join(cmd)}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
    )

    if rc != 0:
        result = RunResult(commit, None, None, wall, "crash", description)
        _append_result(result)
        print(f"crash (rc={rc}, wall={wall:.1f}s); see {RUN_LOG}")
        return 1

    blob = _extract_score_blob(stdout)
    if not blob or "local_score" not in blob or "direction" not in blob:
        result = RunResult(commit, None, None, wall, "crash", description)
        _append_result(result)
        print("crash: solution did not print score JSON blob on its last line")
        return 1

    score = float(blob["local_score"])
    direction = str(blob["direction"])
    best, _ = _read_best_score()
    keep = _is_better(score, best, direction)

    if not CANDIDATE_SUBMISSION.exists():
        result = RunResult(commit, score, direction, wall, "crash", description)
        _append_result(result)
        print("crash: solution did not write submission file at --out")
        return 1

    if keep:
        shutil.copyfile(CANDIDATE_SUBMISSION, BEST_SUBMISSION)
        result = RunResult(commit, score, direction, wall, "keep", description)
        _append_result(result)
        print(
            f"keep: {score:.6f} ({direction}); previous best {best}; "
            f"wall={wall:.1f}s; submission updated"
        )
        return 0
    else:
        # Roll back the experimental commit so the branch tracks the
        # best-known state. Only safe if we're not on the initial commit.
        nparents = int(_git("rev-list", "--count", "HEAD") or "0")
        if nparents > 1:
            _git("reset", "--hard", "HEAD~1")
        result = RunResult(commit, score, direction, wall, "discard", description)
        _append_result(result)
        print(
            f"discard: {score:.6f} ({direction}); best {best}; "
            f"wall={wall:.1f}s; rolled back"
        )
        return 0


def cmd_best(args: argparse.Namespace) -> int:
    best, direction = _read_best_score()
    if best is None:
        print("no kept experiments yet")
        return 0
    print(f"best local_score = {best} ({direction})")
    return 0


def _append_result(result: RunResult) -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER + "\n")
    with RESULTS_TSV.open("a") as fh:
        fh.write(result.to_tsv_row() + "\n")


# ---------------------------------------------------------------------------
# Default solution template
# ---------------------------------------------------------------------------

_DEFAULT_SOLUTION_TEMPLATE = '''"""Stub solution for an MLE-Bench task.

The autoresearch_MLE loop expects this script to:

1. Accept ``--data-dir`` (input) and ``--out`` (where to write the
   submission CSV).
2. Print a single JSON blob on its last line of stdout, of the form
   ``{"local_score": <float>, "direction": "min"|"max", ...}``.

The first iteration's job is to read /home/data/description.md, understand
the task, and replace this stub with a real baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.monotonic()

    # TODO: replace with a real baseline. For now, copy sample_submission.csv
    # if it exists, otherwise write an empty file. The agent loop only
    # advances the branch when ``local_score`` improves, so this stub is
    # guaranteed to be discarded as soon as a real model produces a score.
    sample = args.data_dir / "sample_submission.csv"
    if sample.exists():
        args.out.write_bytes(sample.read_bytes())
    else:
        args.out.write_text("")

    payload = {
        "local_score": 0.0,
        "direction": "max",
        "n_train": 0,
        "n_val": 0,
        "wall_seconds": time.monotonic() - t0,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent.loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="set up /home/code, git repo, results.tsv")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="run a single experiment iteration")
    p_run.add_argument(
        "--description",
        default=None,
        help="short description of the experiment (defaults to last commit subject)",
    )
    p_run.set_defaults(func=cmd_run)

    p_best = sub.add_parser("best", help="print the best kept score so far")
    p_best.set_defaults(func=cmd_best)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
