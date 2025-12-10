"""Base classes for Conformal Prediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt  # noqa: F401


class ConformalPredictor(ABC):
    """Abstract base class for Conformal Predictors."""

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        """Initialize the ConformalPredictor with a model."""
        self.model = model
        self.is_calibrated = False
        self.threshold: float | None = None

    def calibrate(self, x_cal: Any, y_cal: Any, alpha: float) -> None:  # noqa: ANN401, ARG002
        """Placeholder calibration method."""
        # Dummy Implementation
        self.is_calibrated = True
        self.threshold = 0.1  # Dummy threshold

    @abstractmethod
    def predict(self, x: Any, significance_level: float) -> list[Any]:  # noqa: ANN401
        """Abstract predict method."""

    @abstractmethod
    def _compute_nonconformity(self, x: Any, y: Any) -> Any:  # noqa: ANN401
        pass
