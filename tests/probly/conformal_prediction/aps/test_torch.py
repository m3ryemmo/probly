"""Tests for torch APS."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from probly.conformal_prediction.aps.methods.split_conformal import SplitConformal
from probly.conformal_prediction.aps.torch import APSPredictor


class SimpleNet(nn.Module):
    """A simple neural network for testing."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the simple neural network.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model() -> nn.Module:
    """Ficture to create a simple PyTorch model."""
    return SimpleNet(input_dim=5, output_dim=3)


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate dummy data for testing."""
    rng = np.random.default_rng(42)

    # train data
    x_train = rng.random((100, 5), dtype=np.float32)
    y_train = rng.integers(0, 3, size=100)

    # calibration data
    x_calib = rng.random((50, 5), dtype=np.float32)
    y_calib = rng.integers(0, 3, size=50)

    return x_train, y_train, x_calib, y_calib


class TestAPSPredictorTorch:
    """Test class for APSPredictor using PyTorch."""

    def test_initialization(self, simple_model: nn.Module) -> None:
        """Test initialization of APSPredictor."""
        predictor = APSPredictor(model=simple_model)

        assert predictor.model is simple_model
        assert not predictor.is_calibrated
        assert predictor.threshold is None
        assert predictor.nonconformity_scores is None

        # check device
        assert predictor.device.type in ["cpu", "cuda"]

    def test_initialization_with_device(self, simple_model: nn.Module) -> None:
        """Test initialization of APSPredictor with specified device."""
        predictor = APSPredictor(model=simple_model, device="cpu")
        assert predictor.device.type == "cpu"

    def test_calibration(self, simple_model: nn.Module) -> None:
        """Test calibration functionality."""
        predictor = APSPredictor(model=simple_model)

        # create calibration data
        rng = np.random.default_rng(42)
        x_calib = rng.random((30, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=30)

        # calibrate
        significance = 0.1
        threshold = predictor.calibrate(x_calib, y_calib, significance)

        assert predictor.is_calibrated
        assert predictor.threshold == threshold
        assert predictor.nonconformity_scores is not None
        assert len(predictor.nonconformity_scores) == len(x_calib)
        assert 0 <= predictor.threshold <= 1

    def test_prediction_after_calibration(self, simple_model: nn.Module) -> None:
        """Test prediction functionality after calibration."""
        predictor = APSPredictor(model=simple_model)

        # calibrate
        rng = np.random.default_rng(42)
        x_calib = rng.random((20, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=20)
        predictor.calibrate(x_calib, y_calib, significance=0.1)

        # predict
        x_test = rng.random((5, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, significance=0.1)

        # check predictions
        assert len(prediction_sets) == len(x_test)
        assert all(isinstance(pred_set, list) for pred_set in prediction_sets)
        assert all(len(pred_set) >= 1 for pred_set in prediction_sets)

    def test_predictsets_are_valid(self, simple_model: nn.Module) -> None:
        """Test that prediction sets are valid."""
        predictor = APSPredictor(model=simple_model)

        # calibrate
        rng = np.random.default_rng(42)
        x_calib = rng.random((25, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=25)
        predictor.calibrate(x_calib, y_calib, significance=0.1)

        # predict
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, significance=0.1)

        # verify all sets contain valid class indices (0, 1 or 2)
        for pred_set in prediction_sets:
            assert all(0 <= idx < 3 for idx in pred_set)

    def test_str_representation(self, simple_model: nn.Module) -> None:
        """Test string representation of APSPredictor."""
        predictor = APSPredictor(model=simple_model)

        # before calibration
        assert "not calibrated" in str(predictor)
        assert simple_model.__class__.__name__ in str(predictor)

        # after calibration
        rng = np.random.default_rng(42)
        x_calib = rng.random((10, 5), dtype=np.float32)
        y_calib = rng.integers(0, 3, size=10)
        predictor.calibrate(x_calib, y_calib, significance=0.1)

        assert "calibrated" in str(predictor)

    def test_integration_with_split_conformal(self, simple_model: nn.Module) -> None:
        """Test integration with split conformal."""
        predictor = APSPredictor(model=simple_model)

        # create full dataset
        rng = np.random.default_rng(42)
        x_full = rng.random((150, 5), dtype=np.float32)
        y_full = rng.integers(0, 3, size=150)

        # use fit_with_split
        splitter = SplitConformal(calibration_ratio=0.3, random_state=42)

        predictor.set_splitter(splitter)

        x_train, y_train = predictor.fit_with_split(
            x_full,
            y_full,
            significance_level=0.1,
            calibration_ratio=0.3,
        )

        # verify split
        assert predictor.nonconformity_scores is not None
        assert len(x_train) + len(predictor.nonconformity_scores) == len(x_full)
        assert predictor.is_calibrated
        assert predictor.threshold is not None

        # make predictions
        x_test = rng.random((10, 5), dtype=np.float32)
        prediction_sets = predictor.predict(x_test, significance=0.1)

        assert len(prediction_sets) == len(x_test)
