"""Common for SAPS scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.methods.common import Predictor, predict_probs


@lazydispatch
def saps_score_func(
    probs: npt.NDArray[np.floating],
    label: int,
    lambda_val: float = 0.1,
    u: float | None = None,
) -> float:
    """Compute SAPS Nonconformity Score for specific label (Reference: Eq 10).

    prob: 1D array with probabilities.
    label: true index
    lambda_val: lambda value for SAPS.
    u: optional random value in [0,1).
    """
    probs_np = np.asarray(probs, dtype=float)

    if probs_np.ndim == 2:
        if probs_np.shape[0] != 1:
            raise ValueError
        probs_np = probs_np[0]

    if not (0 <= label < probs_np.shape[0]):
        raise ValueError

    if u is None:
        u = float(np.random.Generator(0, 1))

    max_prob = float(np.max(probs_np))
    sorted_indices = np.argsort(-probs_np)
    rank = int(np.where(sorted_indices == label)[0][0]) + 1  # 1-based rank

    if rank == 1:
        return u * max_prob
    return max_prob + (rank - 2 + u) * lambda_val


def register(cls: lazydispatch, func: Callable) -> None:
    """Register an implementation for a specific type."""
    saps_score_func.register(cls=cls, func=func)


# batch helper function for convenience
def saps_score_func_batch(
    probs: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    lambda_val: float = 0.1,
    us: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Batch version of SAPS Nonconformity Score."""
    probs_np = np.asarray(probs, dtype=float)
    n_samples = probs_np.shape[0]

    if us is None:
        us = np.random.Generator(0, 1, size=n_samples)

    scores = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        scores[i] = saps_score_func(
            probs_np[i],
            labels[i],
            lambda_val=lambda_val,
            u=us[i],
        )
    return scores


class SAPSScore:
    """Sorted Adaptive Prediction Sets (SAPS) nonconformity score."""

    def __init__(self, model: Predictor, lambda_val: float = 0.1, random_state: int | None = None) -> None:
        """Initialize SAPS score."""
        self.model = model
        self.lambda_val = lambda_val
        self.rng = np.random.default_rng(random_state)

    def calibration_nonconformity(
        self,
        x_calib: Sequence[Any],
        y_calib: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores for calibration data."""
        probs: npt.NDArray[np.floating]
        probs = predict_probs(self.model, x_calib)
        labels_np = np.asarray(y_calib, dtype=int)

        us = self.rng.uniform(0, 1, size=len(labels_np))

        scores = np.empty(len(labels_np), dtype=float)
        for i in range(len(labels_np)):
            scores[i] = saps_score_func(
                probs=probs[i],
                label=labels_np[i],
                lambda_val=self.lambda_val,
                u=us[i],
            )
        return scores

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute scores for all labels."""
        if probs is None:
            probs = predict_probs(self.model, x_test)

        n_samples, n_classes = probs.shape
        scores = np.empty((n_samples, n_classes), dtype=float)

        for i in range(n_samples):
            for label in range(n_classes):
                us = self.rng.uniform(0, 1, size=n_classes)
                scores[i, label] = saps_score_func(
                    probs=probs[i],
                    label=label,
                    lambda_val=self.lambda_val,
                    u=us[label],
                )

        return scores
