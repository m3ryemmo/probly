# src/probly/conformal_prediction/methods/class_conditional.py
"""Class-conditional split conformal prediction methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.conformal_prediction.scores.common import Score

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from probly.conformal_prediction.methods.common import ConformalPredictor, Predictor, predict_probs
from probly.conformal_prediction.scores.lac.common import accretive_completion
from probly.conformal_prediction.utils.quantile import calculate_quantile


class ClassConditionalPredictor(ConformalPredictor):
    """Class-conditional split conformal predictor for classification.

    Each class y gets its own threshold q_y, calibrated only on samples with true label y.
    This allows to enforce coverage within each class.
    """

    def __init__(
        self,
        model: Predictor,
        score: Score,
        use_accretive: bool = False,
    ) -> None:
        """Create a class-conditional split conformal predictor.

        Args:
            model: base predictive model
            score: nonconformity score object
            use_accretive: whether to apply accretive completion to avoid empty sets.
        """
        super().__init__(model=model)
        self.score = score
        self.use_accretive = use_accretive

        # class_id -> threshold
        self.class_thresholds: npt.NDArray[np.floating] | None = None

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate class-wise thresholds on a calibration dataset."""
        # true label nonconformity scores, shape (n_cal,)
        nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)

        # ensure numpy array
        if torch is not None and isinstance(nonconformity_scores, Tensor):
            scores_np = nonconformity_scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(nonconformity_scores, dtype=float)

        # get true labels as numpy array
        labels_np = np.asarray(y_cal, dtype=int)

        if labels_np.shape[0] != scores_np.shape[0]:
            msg = f"Labels and scores must have same length, got {labels_np.shape[0]} vs {scores_np.shape[0]}"
            raise ValueError(msg)

        # determine number of classes
        n_classes = labels_np.max() + 1
        class_thresholds = np.empty(n_classes, dtype=float)

        # calibrate threshold for each class independently
        for c in range(n_classes):
            mask = labels_np == c
            scores_c = scores_np[mask]

            if scores_c.size == 0:
                # fallback: use a very high threshold (allow all)
                class_thresholds[c] = np.inf
            else:
                class_thresholds[c] = calculate_quantile(scores_c, alpha)

        self.class_thresholds = class_thresholds
        self.threshold = None  # no single global threshold
        self.is_calibrated = True
        return alpha

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
        probs: Any = None,  # noqa: ANN401
    ) -> npt.NDArray[np.bool_]:
        """Return class-conditional prediction sets as (n_instances, n_labels) bool-matrix."""
        if not self.is_calibrated or self.class_thresholds is None:
            msg = "Predictor must be calibrated before predict()."
            raise RuntimeError(msg)

        # scores for all labels, shape (n_test, n_labels)
        scores = self.score.predict_nonconformity(x_test)

        if scores.ndim != 2:
            msg = "predict_nonconformity must return 2D-Matrix (n_instances, n_labels)."
            raise ValueError(msg)

        if torch is not None and isinstance(scores, Tensor):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores, dtype=float)

        n_test, n_labels = scores_np.shape

        # class_thresholds shape (n_classes,)
        thresholds = self.class_thresholds[None, :]

        if thresholds.shape[1] != n_labels:
            msg = (
                f"Number of labels in test data ({n_labels}) does not match "
                f"number of classes in calibration ({thresholds.shape[1]})"
            )
            raise ValueError(msg)

        # label y is included iff score(x,y) <= threshold_y
        # compare scores to thresholds
        prediction_sets = scores_np <= thresholds  # (n_test, n_labels)

        if self.use_accretive:
            if probs is None:
                probs = predict_probs(self.model, x_test)
            if torch is not None and isinstance(probs, Tensor):
                probs = probs.detach().cpu().numpy()
            probs_np = np.asarray(probs)
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets
