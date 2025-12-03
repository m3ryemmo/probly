"""Implement common utilities and methods for conformal prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt


class PredictiveModel(Protocol):
    """Protocol for models used with ConformalPredictor."""

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Predict method signature for conformal models."""


class ConformalPredictor(ABC):
    """base class for Conformal Prediction."""

    def __init__(
        self,
        model: PredictiveModel,
        nonconformity_func: Callable[..., npt.NDArray[np.floating]] | None = None,
    ) -> None:
        """Initialze the Conformal Predictor."""
        self.model = model
        self.conformity_func = nonconformity_func
        """saves the ML-model and nonconformity function"""
        self.nonconformity_scores: npt.NDArray[np.floating] | None = None
        self.threshold: float | None = None
        self.is_calibrated: bool = False

    @abstractmethod
    def _compute_nonconformity(self, x: Sequence[Any], y: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores for given data."""

    @abstractmethod
    def predict(self, x: Sequence[Any], significance_level: float) -> Sequence[Any]:
        """Generate prediction sets for given data at specified significance level."""

    def calibrate(self, x_cal: Sequence[Any], y_cal: Sequence[Any], significance_level: float) -> float | None:
        """Calibrate the conformal predictor using calibration data."""
        self.nonconformity_scores = self._compute_nonconformity(x_cal, y_cal)
        """Stores nonconformity scores for later use in prediction."""

        alpha = significance_level
        self.threshold = float(np.quantile(self.nonconformity_scores, 1 - alpha))
        """Computes and stores the threshold for the given significance level."""

        self.is_calibrated = True

        return self.threshold

    # noch splitter methode einfügen, warten bis gemerged wurde
