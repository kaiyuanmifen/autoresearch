"""Submission helpers for autoresearch_MLE solutions.

These are imported by ``solution.py`` (which the agent edits) so the
common bookkeeping -- checking the schema against ``sample_submission.csv``
and writing the file atomically -- doesn't need to be re-implemented in
every iteration.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is in the base image
    pd = None  # type: ignore[assignment]


def expected_columns(data_dir: Path) -> list[str] | None:
    """Return the expected submission columns from sample_submission.csv.

    Returns None if no sample submission exists -- some MLE-Bench tasks
    don't ship one and the agent has to infer the schema from the task
    description.
    """
    sample = Path(data_dir) / "sample_submission.csv"
    if not sample.exists() or pd is None:
        return None
    return list(pd.read_csv(sample, nrows=0).columns)


def write_submission(df, out: Path, data_dir: Path | None = None) -> None:
    """Write ``df`` to ``out`` atomically, validating against the sample.

    Validation: if a sample submission exists, ``df``'s columns must match
    it exactly (same names, same order). The check is strict on purpose --
    a CSV with the right rows but wrong column order is a silent zero on
    most Kaggle metrics.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if data_dir is not None:
        expected = expected_columns(Path(data_dir))
        if expected is not None:
            actual = list(df.columns)
            if actual != expected:
                raise ValueError(
                    f"submission columns {actual!r} do not match "
                    f"sample_submission.csv {expected!r}"
                )

    fd, tmp_path = tempfile.mkstemp(
        prefix=".submission-", suffix=".csv", dir=str(out.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def chunked(seq: Iterable, n: int) -> Iterable[list]:
    """Yield successive n-sized chunks from ``seq``. Convenience for batched IO."""
    buf: list = []
    for item in seq:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
