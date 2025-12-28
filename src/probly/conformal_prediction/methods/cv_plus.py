"""Cross-validation+ (CV+) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from probly.conformal_prediction.scores.aps.common import APSScore
from probly.conformal_prediction.scores.lac.common import LACScore, accretive_completion

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from probly.conformal_prediction.methods.common import PredictiveModel


class CVPlus:
    """Cross-validation+ (CV+) conformal predictor."""

    def __init__(
        self,
        score_type: Literal["lac", "aps"] = "aps",
        model_factory: Callable[[], Any] | None = None,
        n_splits: int = 5,
        random_state: int | None = None,
        alpha: float = 0.1,
    ) -> None:
        """Initialize CV+ conformal predictor.

        Args:
            score_type: Type of nonconformity score ("lac" or "aps").
            model_factory: A model factory function that returns a new model instance.
            n_splits: Number of folds for cross-validation.
            random_state: Random seed for reproducibility.
            alpha: Significance level for prediction sets.
        """
        self.score_type = score_type
        self.model_factory = model_factory
        self.n_splits = n_splits
        self.random_state = random_state
        self.alpha = alpha

        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()

        self.models: list[Any] = []
        self.calibration_scores: list[np.ndarray] = []
        self.fold_assignments: np.ndarray | None = None

    def _get_score_object(self, model: PredictiveModel) -> LACScore | APSScore:
        """Get score object based on score_type."""
        if self.score_type == "aps":
            return APSScore(model)
        return LACScore(model)

    def fit(self, x: Sequence[Any], y: Sequence[Any]) -> CVPlus:
        """Fit CV+ conformal predictor."""
        x_array = np.asarray(x)
        y_array = np.asarray(y, dtype=int)
        n_samples = len(x)

        # assign folds
        idx = self.rng.permutation(n_samples) if self.random_state else np.arange(n_samples)
        self.fold_assignments = np.zeros(n_samples, dtype=int)
        fold_size = n_samples // self.n_splits

        for k in range(self.n_splits):
            start = k * fold_size
            end = (k + 1) * fold_size if k < self.n_splits - 1 else n_samples
            self.fold_assignments[idx[start:end]] = k

        # train model and collect scores
        self.models = []
        self.calibration_scores = []

        for k in range(self.n_splits):
            train_mask = self.fold_assignments != k
            x_train, y_train = x_array[train_mask], y_array[train_mask]

            model = self.model_factory()
            model.fit(x_train, y_train)
            self.models.append(model)

            # get calibration scores
            cal_mask = self.fold_assignments == k
            if np.any(cal_mask):
                score_obj = self._get_score_object(model)
                scores = score_obj.calibration_nonconformity(
                    x_array[cal_mask],
                    y_array[cal_mask],
                )
                self.calibration_scores.append(scores)

        return self

    def predict(self, x_test: Sequence[Any]) -> np.ndarray:
        """Predict conformal sets."""
        x_test_array = np.asarray(x_test)
        n_test = len(x_test_array)

        # infer number of classes
        test_probs = self.models[0].predict(x_test_array[:1])
        n_classes = test_probs.shape[1]

        # initialize prediction sets
        all_scores = np.zeros((self.n_splits, n_test, n_classes))
        prediction_sets = np.zeros((n_test, n_classes), dtype=bool)

        # calculate scores for all folds
        for k, model in enumerate(self.models):
            score_obj = self._get_score_object(model)

            # scores for all possible labels
            for c in range(n_classes):
                y_cand = np.full(n_test, c, dtype=int)
                all_scores[k, :, c] = score_obj.calibration_nonconformity(
                    x_test_array,
                    y_cand,
                )

        # cv+ rule
        total_cal = sum(len(s) for s in self.calibration_scores)
        threshold = (1 - self.alpha) * (total_cal + 1)

        for i in range(n_test):
            for c in range(n_classes):
                count = sum(np.sum(s < all_scores[k, i, c]) for k, s in enumerate(self.calibration_scores))
                if count < threshold:
                    prediction_sets[i, c] = True

        if self.score_type == "lac":
            avg_probs = np.mean([m.predict(x_test_array) for m in self.models], axis=0)

            score_comp = 1.0 - avg_probs
            prediction_sets = accretive_completion(prediction_sets, score_comp)

        return prediction_sets

    def predict_with_probs(self, x_test: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Predict conformal sets with averaged probabilities."""
        x_test_array = np.asarray(x_test)
        avg_probs = np.mean([m.predict(x_test_array) for m in self.models], axis=0)
        sets = self.predict(x_test_array)
        return sets, avg_probs

    def get_coverage(self, x_test: Sequence[Any], y_test: Sequence[Any]) -> tuple[float, float]:
        """Calculate coverage metrics on test data."""
        x_test_array = np.asarray(x_test)
        y_test_array = np.asarray(y_test, dtype=int)
        sets = self.predict(x_test_array)

        # marginal coverage
        marginal = np.mean([sets[np.arange(len(y_test_array)), y_test_array]])

        # conditional coverage
        unique_labels = np.unique(y_test_array)
        conditional_coverages = []

        for label in unique_labels:
            mask = y_test_array == label
            if np.sum(mask) > 0:
                label_covered = np.mean(sets[mask, label])
                conditional_coverages.append(label_covered)

        conditional = np.mean(conditional_coverages) if conditional_coverages else 0.0

        return marginal, conditional
