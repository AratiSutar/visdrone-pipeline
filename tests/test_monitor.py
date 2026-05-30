import pytest
import numpy as np
from src.drift.monitor import kl_divergence, compute_drift_score


def test_kl_divergence_identical():
    """KL divergence of identical distributions should be near 0."""
    p = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([1.0, 2.0, 3.0, 4.0])
    score = kl_divergence(p, q)
    assert score < 0.01


def test_kl_divergence_different():
    """KL divergence of different distributions should be > 0."""
    p = np.array([1.0, 1.0, 1.0, 1.0])
    q = np.array([4.0, 1.0, 1.0, 1.0])
    score = kl_divergence(p, q)
    assert score > 0


def test_kl_divergence_non_negative():
    """KL divergence should always be non negative."""
    p = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([4.0, 3.0, 2.0, 1.0])
    score = kl_divergence(p, q)
    assert score >= 0


def test_compute_drift_score_identical():
    """Drift score should be low for identical distributions."""
    confs = [0.8, 0.7, 0.9, 0.6, 0.85]
    score = compute_drift_score(confs, confs)
    assert score < 0.01


def test_compute_drift_score_different():
    """Drift score should be higher for different distributions."""
    baseline = [0.8, 0.9, 0.85, 0.75, 0.9]
    drifted  = [0.2, 0.3, 0.25, 0.15, 0.2]
    score = compute_drift_score(baseline, drifted)
    assert score > 0.1


def test_compute_drift_score_empty():
    """Drift score with empty list should not crash."""
    score = compute_drift_score([], [0.5, 0.6])
    assert score >= 0