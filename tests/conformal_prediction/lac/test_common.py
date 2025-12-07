from __future__ import annotations

from typing import Any

import numpy as np

from probly.conformal_prediction.lac.common import (
    LAC,
    accretive_completion,
    calculate_local_weights,
    calculate_non_conformity_score,
    calculate_weighted_quantile,
)


class MockModel:
    """A dummy model to simulate sklearn's predict/predict_proba."""

    def predict(self, x: Any) -> Any:  # noqa: ARG002, ANN401
        """Return fixed probabilities for testing."""
        # Output shape is (n_samples, n_classes)
        return np.array(
            [
                [0.1, 0.8, 0.1],
                [0.6, 0.3, 0.1],
                [0.2, 0.2, 0.6],
            ],
        )


def test_calculate_non_conformity_score() -> None:
    """Test calculation of non-conformity scores: 1 - p(y|x).

    Based on LAC logic.
    """
    probas = np.array(
        [
            [0.1, 0.8, 0.1],  # True class 1 -> Score 1-0.8 = 0.2
            [0.6, 0.3, 0.1],  # True class 0 -> Score 1-0.6 = 0.4
            [0.2, 0.2, 0.6],  # True class 2 -> Score 1-0.6 = 0.4
        ],
    )
    y_true = np.array([1, 0, 2])

    scores = calculate_non_conformity_score(probas, y_true)

    expected = np.array([0.2, 0.4, 0.4])
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_accretive_completion() -> None:
    """Test that empty sets are filled with the class of highest probability.

    See Section 3.2 'Accretive Completion' in the paper.
    """
    # Sample 0: Empty set -> Should pick index 1 (0.4 prob)
    # Sample 1: Not empty -> Should stay same
    prediction_sets = np.array(
        [
            [False, False, False],
            [True, False, False],
        ],
    )
    # Probabilities corresponding to the sets
    probas = np.array(
        [
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ],
    )

    result = accretive_completion(prediction_sets, probas)

    expected = np.array(
        [
            [False, True, False],  # Filled!
            [True, False, False],  # Unchanged
        ],
    )
    np.testing.assert_array_equal(result, expected)


def test_calculate_weighted_quantile() -> None:
    """Test standard quantile calculation required by goals."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Median without weights
    q = calculate_weighted_quantile(values, 0.5, sample_weight=None)
    assert q == 3.0


def test_calculate_local_weights() -> None:
    """Test that weights are returned as ones (default for LAC)."""
    x = np.zeros((5, 3))
    weights = calculate_local_weights(x)
    assert weights.shape == (5,)
    assert np.all(weights == 1.0)


def test_lac_class_integration() -> None:
    """Test the LAC class flow."""
    model = MockModel()
    predictor = LAC(model)

    # 1. Test non-conformity computation using the internal method
    x_dummy = np.zeros((3, 5))
    y_dummy = np.array([1, 0, 2])

    scores = predictor._compute_nonconformity(x_dummy, y_dummy)  # noqa: SLF001

    # Expect 1 - p(y_true)
    expected_scores = np.array([0.2, 0.4, 0.4])
    np.testing.assert_allclose(scores, expected_scores, atol=1e-6)

    # 2. Test predict (Mocking calibration state)
    predictor.is_calibrated = True
    predictor.threshold = 0.5  # Corresponds to prob_threshold = 1 - 0.5 = 0.5

    # Model Probas:
    # [0.1, 0.8, 0.1] -> Class 1 > 0.5
    # [0.6, 0.3, 0.1] -> Class 0 > 0.5
    # [0.2, 0.2, 0.6] -> Class 2 > 0.5

    res = predictor.predict(x_dummy, significance_level=0.1)

    # Check if we got list of arrays
    assert isinstance(res, list)
    assert len(res) == 3
    # Use direct boolean check instead of '== True'
    assert res[0][1]  # Class 1 selected
    assert res[1][0]  # Class 0 selected
