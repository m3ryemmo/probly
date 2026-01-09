"""Cross-validation+ (CV+) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from probly.conformal_prediction.methods.common import Predictor

from probly.conformal_prediction.methods.common import ConformalPredictor, predict_probs


class CrossValidationPredictor(ConformalPredictor):
    """Cross-validation+ (CV+) conformal predictor."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        task: Literal["classification", "regression"] = "classification",
        n_splits: int = 5,
        random_state: int | None = None,
    ) -> None:
        """Initialize CV+ conformal predictor.

        Args:
            model_factory: A callable that returns a new instance of the model.
            task: The type of task, either "classification" or "regression".
            n_splits: Number of folds for cross-validation.
            random_state: Random seed for reproducibility.

        """
        super().__init__(model=model_factory())
        self.model_factory = model_factory
        self.task = task
        self.n_splits = n_splits
        self.rng = np.random.default_rng(random_state)

        self.models: list[Predictor] = []
        self.calibration_scores: list[npt.NDArray[np.floating]] = []

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate the CV+ conformal predictor on the calibration data.

        Args:
            x_cal: Calibration input data.
            y_cal: Calibration labels.
            alpha: Significance level for prediction sets.

        Returns:
            The threshold used for prediction sets.
        """
        x = np.asarray(x_cal)
        y = np.asarray(y_cal)
        n_samples = len(x)

        # create folds
        idx = self.rng.permutation(n_samples)
        fold_size = n_samples // self.n_splits

        for k in range(self.n_splits):
            start = k * fold_size
            end = (k + 1) * fold_size if k < self.n_splits - 1 else n_samples
            cal_idx = idx[start:end]
            train_idx = np.setdiff1d(idx, cal_idx)

            model = self.model_factory()
            if hasattr(model, "fit"):
                model.fit(x[train_idx], y[train_idx])
            else:
                raise TypeError
            self.models.append(model)

            # Scores
            if self.task == "classification":
                probs: npt.NDArray[np.floating] = predict_probs(model, x[cal_idx])
                row_idx = np.arange(len(cal_idx))
                # LAC
                scores = 1.0 - probs[row_idx, y[cal_idx]]
            else:
                # Regression
                preds: npt.NDArray[np.floating] = predict_probs(model, x[cal_idx])
                if preds.ndim > 1:
                    preds = preds.ravel()
                scores = np.abs(y[cal_idx] - preds)

            self.calibration_scores.append(scores)

        self.is_calibrated = True

        all_scores = np.concatenate(self.calibration_scores)
        return float(np.quantile(all_scores, 1 - alpha, method="higher"))

    def predict(self, x_test: Sequence[Any], alpha: float) -> npt.NDArray[Any]:
        """Predict condormal sets.

        Args:
          x_test: Test input data.
          alpha: Significance level for prediction sets.

        Returns:
          For classification: A boolean array of shape (n_instances, n_labels).
          For regression: An array of shape (n_instances, 2) with lower and upper bounds.
        """
        if not self.is_calibrated:
            raise RuntimeError

        x = np.asarray(x_test)

        if self.task == "classification":
            # Get all predictions
            all_probs = np.array([predict_probs(m, x) for m in self.models])
            avg_probs = np.mean(all_probs, axis=0)

            # LAC threshold
            all_cal = np.concatenate(self.calibration_scores)
            quantile = np.quantile(all_cal, 1 - alpha, method="higher")

            return avg_probs > (1 - quantile)

        # Regression intervals
        all_preds: list[npt.NDArray[np.floating]] = []
        for m in self.models:
            pred: npt.NDArray[np.floating] = predict_probs(m, x)
            if pred.ndim > 1:
                pred = pred.ravel()
            all_preds.append(pred)

        avg_pred = np.mean(all_preds, axis=0)

        all_res = np.concatenate(self.calibration_scores)
        quantile = np.quantile(all_res, 1 - alpha, method="higher")

        return np.column_stack([avg_pred - quantile, avg_pred + quantile])
