"""Mondrian split conformal prediction methods."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from probly.conformal_prediction.scores.common import Score

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from probly.conformal_prediction.methods.common import ConformalPredictor, Predictor, predict_probs
from probly.conformal_prediction.scores.lac.common import accretive_completion
from probly.conformal_prediction.utils.quantile import calculate_quantile

# region_func(x) -> int-array of region/group ids
RegionFunc = Callable[[Sequence[Any]], npt.NDArray[np.int_]]


class MondrianConformalPredictor(ConformalPredictor):
    """Mondrian (region-wise) split conformal predictor for classification.

    Regions are defined by `region_func(x)`. Each region gets its own threshold.
    This allows to enforce coverage within each region, useful when different regions
    have different difficulty levels.
    """

    def __init__(
        self,
        model: Predictor,
        score: Score,
        region_func: RegionFunc,
        use_accretive: bool = False,
    ) -> None:
        """Create a Mondrian split conformal predictor.

        Args:
            model: base predictive model
            score: nonconformity score object
            region_func: function mapping input x to region/group ids (ints).
            use_accretive: whether to apply accretive completion to avoid empty sets.
        """
        super().__init__(model=model)
        self.score = score
        self.region_func = region_func
        self.use_accretive = use_accretive

        # region_id -> threshold
        self.region_thresholds: dict[int, float] = {}

    def calibrate(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        alpha: float,
    ) -> float:
        """Calibrate region-wise thresholds on a calibration dataset."""
        # true label nonconformity scores, shape (n_cal,)
        nonconformity_scores = self.score.calibration_nonconformity(x_cal, y_cal)

        # ensure numpy array
        if torch is not None and isinstance(nonconformity_scores, Tensor):
            scores_np = nonconformity_scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(nonconformity_scores, dtype=float)

        # compute regions for each calibration instance
        regions = self.region_func(x_cal)
        regions_np = np.asarray(regions, dtype=int)

        if regions_np.shape[0] != scores_np.shape[0]:
            msg = (
                "region_func must return one region id per calibration instance, "
                f"got {regions_np.shape[0]} vs {scores_np.shape[0]}"
            )
            raise ValueError(msg)

        self.region_thresholds.clear()
        unique_regions = np.unique(regions_np)

        for g in unique_regions:
            mask = regions_np == g
            scores_g = scores_np[mask]
            if scores_g.size == 0:
                continue
            # standard split-conformal quantile per region
            self.region_thresholds[int(g)] = calculate_quantile(scores_g, alpha)

        self.is_calibrated = True
        self.threshold = None  # no single global threshold
        return alpha

    def predict(
        self,
        x_test: Sequence[Any],
        alpha: float,  # noqa: ARG002
        probs: Any = None,  # noqa: ANN401
    ) -> npt.NDArray[np.bool_]:
        """Return Mondrian prediction sets as (n_instances, n_labels) bool-matrix."""
        if not self.is_calibrated or not self.region_thresholds:
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

        n_test, _ = scores_np.shape

        # compute regions for each test instance
        regions = self.region_func(x_test)
        regions_np = np.asarray(regions, dtype=int)

        if regions_np.shape[0] != n_test:
            msg = f"region_func must return one region id per test instance, got {regions_np.shape[0]} vs {n_test}"
            raise ValueError(msg)

        # threshold per test sample: q_{region(x_i)}
        thresholds_per_sample = np.empty(n_test, dtype=float)
        max_threshold = max(self.region_thresholds.values())

        for i, g in enumerate(regions_np):
            g_int = int(g)
            # fallback: if region not seen in calibration, use max threshold (most conservative)
            thresholds_per_sample[i] = self.region_thresholds.get(g_int, max_threshold)

        thresholds_per_sample = thresholds_per_sample.reshape(n_test, 1)  # (n_test, 1)

        # compare scores to thresholds
        prediction_sets = scores_np <= thresholds_per_sample  # (n_test, n_labels)

        if self.use_accretive:
            if probs is None:
                probs = predict_probs(self.model, x_test)
            if torch is not None and isinstance(probs, Tensor):
                probs = probs.detach().cpu().numpy()
            probs_np = np.asarray(probs)
            prediction_sets = accretive_completion(prediction_sets, probs_np)

        return prediction_sets
