"""Tests for the Flax implementation of LAC."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from probly.conformal_prediction.lac.flax import LACFlax


class MockFlaxModel:
    """Simulates a Flax module."""

    def apply(self, params: Any, x: Any) -> Any:  # noqa: ANN401, ARG002
        """Return fixed logits regardless of input x.

        Using logits that result in known probabilities after softmax.
        """
        n_samples = x.shape[0]
        # Logits: [1.0, 0.5, 0.2] -> Softmax roughly [0.5, 0.3, 0.2]
        logits = jnp.array([[1.0, 0.5, 0.2]])
        # Tile them to match batch size
        return jnp.tile(logits, (n_samples, 1))


def test_flax_prediction_flow() -> None:
    """Test that LACFlax runs through predict pipeline."""
    model = MockFlaxModel()
    params: dict[str, Any] = {}  # Dummy params
    predictor = LACFlax(model, params)

    # Fake Calibration
    predictor.is_calibrated = True
    predictor.threshold = 0.8  # High threshold -> Strict
    # Logic: Score s = 1 - p. Threshold q.
    # Prediction set included if: 1-p <= q  <=>  p >= 1-q
    # 1 - 0.8 = 0.2. So any class with prob >= 0.2 should be included.

    x_dummy = np.zeros((2, 5))  # 2 Samples

    # Run predict
    # Since MockModel returns constant logits, we expect valid sets.
    sets = predictor.predict(x_dummy, significance_level=0.1)

    assert len(sets) == 2
    assert isinstance(sets[0], np.ndarray)


def test_flax_nonconformity() -> None:
    """Test calculation of scores using Flax wrapper."""
    model = MockFlaxModel()
    predictor = LACFlax(model, params={})

    x_dummy = np.zeros((3, 5))
    y_dummy = np.array([0, 1, 2])

    scores = predictor._compute_nonconformity(x_dummy, y_dummy)  # noqa: SLF001

    assert scores.shape == (3,)
    # Check that scores are between 0 and 1
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
