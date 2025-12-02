"""Split Conformal Prediction Method Implementation."""

from __future__ import annotations

import numpy as np


class SplitConformal:
    """Implementiert die Split-Conformal Methode.

    This class splits data into training and calibration sets
    for conformal prediction using the split conformal approach.
    The split is done randomly based on a specified calibration ratio.
    """

    def __init__(
        self,
        calibration_ratio: float = 0.3,
        random_state: int | None = None,
    ) -> None:
        """Initialize the SplitConformal class.

        Args:
        calibration_ratio: ratio of data to use for calibration (set to 0.3 by default).
        random_state: for reproducibility of random splits.
        """
        self.calibration_ratio = calibration_ratio
        self.random_state = random_state

        # modern random generator
        self.rng = np.random.default_rng(random_state)

        # save info about last split
        self.last_split_info: dict[str, object] | None = None

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray,
        calibration_ratio: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits data into training and calibration sets.

        Args:
            x: Features
            y: Labels
            calibration_ratio: overrides default if provided

        Returns:
            x_train, y_train, x_cal, y_cal
        """
        # decide which calibration ratio to use (user friendly)
        if calibration_ratio is not None:
            # use the provided value (overrides default)
            ratio_to_use = calibration_ratio
            used_default = False
        else:
            # use the default value
            ratio_to_use = self.calibration_ratio
            used_default = True

        # make sure inputs are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        n_samples = len(x)

        # create shuffled indices
        indices = np.arange(n_samples)
        shuffled_indices = self.rng.permutation(indices)

        # calculate split index with the correct ratio
        split_idx = int(n_samples * (1 - ratio_to_use))

        # split indices
        train_indices = shuffled_indices[:split_idx]
        cal_indices = shuffled_indices[split_idx:]

        # save split info
        self.last_split_info = {
            "train_indices": train_indices,
            "cal_indices": cal_indices,
            "n_training": len(train_indices),
            "n_calibration": len(cal_indices),
            "calibration_ratio_used": ratio_to_use,
            "calibration_ratio_actual": len(cal_indices) / n_samples,
            "used_default": used_default,
            "default_ratio": self.calibration_ratio,
        }

        # give back the splits
        return (
            x[train_indices],
            y[train_indices],
            x[cal_indices],
            y[cal_indices],
        )

    def get_split_info(self) -> dict[str, object]:
        """Gives information about the last split."""
        if self.last_split_info is None:
            return {"status": "no split performed yet"}
        return self.last_split_info

    def __str__(self) -> str:
        """String representation of the class."""
        # get split info
        info = self.get_split_info()
        # check if no split was performed yet
        if "status" in info:
            return f"SplitConformal(ratio={self.calibration_ratio}, random_state={self.random_state})"

        # split was performed, show details
        source = "default" if info["used_default"] else "custom"
        return (
            f"SplitConformal: {info['n_training']} Training, "
            f"{info['n_calibration']} Calibration "
            f"(ratio={info['calibration_ratio_used']}, {source})"
        )
