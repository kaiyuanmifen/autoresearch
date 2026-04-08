"""Unit tests for the pure-function helpers in agent.loop and agent.grading.

These intentionally avoid touching the filesystem or git -- the
end-to-end behaviour is exercised separately by running ``agent.loop``
in a temp directory (see the README).
"""

from __future__ import annotations

import math

import pytest

from agent import grading
from agent.loop import _extract_score_blob, _is_better


# ---------------------------------------------------------------------------
# loop._is_better
# ---------------------------------------------------------------------------


def test_is_better_first_score_always_kept():
    assert _is_better(0.5, None, "max") is True
    assert _is_better(0.5, None, "min") is True


def test_is_better_max_direction():
    assert _is_better(0.6, 0.5, "max") is True
    assert _is_better(0.4, 0.5, "max") is False
    assert _is_better(0.5, 0.5, "max") is False  # equal is not better


def test_is_better_min_direction():
    assert _is_better(0.4, 0.5, "min") is True
    assert _is_better(0.6, 0.5, "min") is False
    assert _is_better(0.5, 0.5, "min") is False


def test_is_better_unknown_direction_raises():
    with pytest.raises(ValueError):
        _is_better(0.5, 0.4, "sideways")


# ---------------------------------------------------------------------------
# loop._extract_score_blob
# ---------------------------------------------------------------------------


def test_extract_score_blob_simple():
    out = '{"local_score": 0.81, "direction": "max"}'
    blob = _extract_score_blob(out)
    assert blob == {"local_score": 0.81, "direction": "max"}


def test_extract_score_blob_picks_last_line():
    out = (
        "loading data...\n"
        "epoch 1 done\n"
        '{"local_score": 0.55, "direction": "min", "n_train": 1000}\n'
    )
    blob = _extract_score_blob(out)
    assert blob is not None
    assert blob["local_score"] == 0.55
    assert blob["direction"] == "min"


def test_extract_score_blob_returns_none_when_missing():
    assert _extract_score_blob("nothing useful here\n") is None
    assert _extract_score_blob("") is None


def test_extract_score_blob_returns_none_on_invalid_json():
    assert _extract_score_blob('{"local_score": not-a-number}') is None


# ---------------------------------------------------------------------------
# grading metrics
# ---------------------------------------------------------------------------


def test_rmse_basic():
    assert grading.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert grading.rmse([0.0, 0.0], [1.0, -1.0]) == pytest.approx(1.0)


def test_mae_basic():
    assert grading.mae([1.0, 2.0], [1.5, 2.5]) == pytest.approx(0.5)


def test_accuracy():
    assert grading.accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    assert math.isnan(grading.accuracy([], []))


def test_log_loss_perfect_predictions_low():
    # near-perfect predictions => near-zero loss
    loss = grading.log_loss([1, 0, 1, 0], [0.99, 0.01, 0.99, 0.01])
    assert loss < 0.05


def test_log_loss_clipping_avoids_infinities():
    # extreme predictions should be clipped, not raise
    loss = grading.log_loss([1, 0], [1.0, 0.0])
    assert math.isfinite(loss)


def test_auc_perfect_separation():
    # all positives ranked above all negatives
    score = grading.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert score == pytest.approx(1.0)


def test_auc_inverted():
    score = grading.auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert score == pytest.approx(0.0)


def test_auc_random_is_half():
    # ties in score => mid rank => AUC = 0.5
    score = grading.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert score == pytest.approx(0.5)


def test_auc_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        grading.auc([0, 1], [0.5])


def test_directions_registry_covers_all_metrics():
    expected = {"rmse", "mae", "log_loss", "accuracy", "auc"}
    assert set(grading.DIRECTIONS) == expected
    assert grading.DIRECTIONS["rmse"] == "min"
    assert grading.DIRECTIONS["accuracy"] == "max"
