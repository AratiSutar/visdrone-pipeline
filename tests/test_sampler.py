import pytest
import numpy as np


def test_uncertainty_score_no_detections():
    """Images with no detections should have maximum uncertainty."""
    # Simulate no detections = uncertainty of 1.0
    confidences = []
    uncertainty = 1.0 if not confidences else 1.0 - float(np.mean(confidences))
    assert uncertainty == 1.0


def test_uncertainty_score_high_confidence():
    """High confidence detections should have low uncertainty."""
    confidences = [0.95, 0.90, 0.92]
    uncertainty = 1.0 - float(np.mean(confidences))
    assert uncertainty < 0.15


def test_uncertainty_score_low_confidence():
    """Low confidence detections should have high uncertainty."""
    confidences = [0.2, 0.15, 0.18]
    uncertainty = 1.0 - float(np.mean(confidences))
    assert uncertainty > 0.8


def test_uncertainty_score_range():
    """Uncertainty score should always be between 0 and 1."""
    confidences = [0.5, 0.6, 0.7]
    uncertainty = 1.0 - float(np.mean(confidences))
    assert 0 <= uncertainty <= 1


def test_sorting_by_uncertainty():
    """Most uncertain images should be first after sorting."""
    scores = [
        {"image": "a.jpg", "uncertainty": 0.3},
        {"image": "b.jpg", "uncertainty": 0.9},
        {"image": "c.jpg", "uncertainty": 0.6},
    ]
    sorted_scores = sorted(scores, key=lambda x: x["uncertainty"], reverse=True)
    assert sorted_scores[0]["image"] == "b.jpg"
    assert sorted_scores[-1]["image"] == "a.jpg"


def test_threshold_filtering():
    """Only images above threshold should be selected."""
    scores = [0.3, 0.7, 0.5, 0.8, 0.4]
    threshold = 0.6
    selected = [s for s in scores if s >= threshold]
    assert len(selected) == 2
    assert all(s >= threshold for s in selected)