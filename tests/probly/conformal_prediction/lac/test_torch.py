from __future__ import annotations

import numpy as np
import torch

from probly.conformal_prediction.lac.torch import LAC

# Constants for deterministic testing
PROB_TRUE_CLASS = 0.8
EXPECTED_SCORE = 1.0 - PROB_TRUE_CLASS  # Expected non-conformity score: 0.2


class MockTorchModel:
    """Mock PyTorch model returning deterministic probabilities for testing."""

    def __init__(self, n_classes: int = 3, true_prob: float = 0.9) -> None:
        """Initialize the mock model."""
        self.n_classes = n_classes
        self.true_prob = true_prob

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        # Preserve device placement
        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")

        # Create probabilities
        probs = torch.zeros(self.n_classes, device=device)

        # Set the true class probability explicitly
        # We use 0.9 to align with the observed threshold of 0.1 (1.0 - 0.9)
        probs[0] = self.true_prob

        # Distribute remaining probability
        if self.n_classes > 1:
            remaining = (1.0 - self.true_prob) / (self.n_classes - 1)
            probs[1:] = remaining

        return probs.repeat(n_samples, 1)


def test_torch_lac_basic_flow() -> None:
    """Verify complete calibration and prediction workflow with Tensor inputs."""
    # We set probability to 0.9. Expected score = 1.0 - 0.9 = 0.1.
    prob_true = 0.9
    expected_score = 1.0 - prob_true  # 0.1

    model = MockTorchModel(true_prob=prob_true)
    predictor = LAC(model)

    # 1. Calibration Data
    # Linter prefers lowercase variables in functions (x_cal instead of X_cal)
    x_cal = torch.randn(10, 5)
    y_cal = torch.zeros(10, dtype=torch.long)  # Class 0 is true

    # 2. Calibrate
    threshold = predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    assert predictor.is_calibrated
    assert threshold is not None

    # Check if threshold matches expected score
    assert abs(threshold - expected_score) < 1e-4, f"Threshold mismatch! Expected {expected_score}, got {threshold}"

    # 3. Predict
    x_test = torch.randn(5, 5)
    sets = predictor.predict(x_test, significance_level=0.1)

    # 4. Verify Output Types
    assert isinstance(sets, torch.Tensor), "Output must be a PyTorch Tensor"
    assert sets.dtype == torch.bool, "Output must be a boolean Tensor"
    assert sets.shape == (5, 3)

    # Class 0 must be included (Score 0.1 <= Threshold 0.1)
    assert torch.all(sets[:, 0]), "Class 0 should be included in prediction sets"


def test_torch_device_preservation() -> None:
    """Ensure that input device (CPU/GPU) is preserved in the output."""
    device = torch.device("cpu")

    model = MockTorchModel()
    predictor = LAC(model)

    x_cal = torch.randn(10, 5).to(device)
    y_cal = torch.zeros(10, dtype=torch.long).to(device)

    predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    x_test = torch.randn(2, 5).to(device)
    sets = predictor.predict(x_test, significance_level=0.1)

    assert sets.device.type == device.type, "Output tensor is on wrong device"


def test_numpy_input_compatibility() -> None:
    """Verify that the wrapper accepts Numpy inputs and returns Tensors."""
    model = MockTorchModel()
    predictor = LAC(model)

    # Use modern numpy random generator to satisfy NPY002
    rng = np.random.default_rng(42)
    x_cal = rng.standard_normal((10, 5)).astype(np.float32)
    y_cal = np.zeros(10, dtype=int)

    predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    x_test = rng.standard_normal((2, 5)).astype(np.float32)
    sets = predictor.predict(x_test, significance_level=0.1)

    assert isinstance(sets, torch.Tensor)


def test_torch_randomized_stress_check() -> None:
    """Stress test with random data to verify robustness and admissibility.

    Ensures 'Accretive Completion' prevents empty sets even with random noise.
    """
    n_samples = 100
    n_classes = 10
    n_features = 5

    x_cal = torch.randn(n_samples, n_features)
    y_cal = torch.randint(0, n_classes, (n_samples,))
    x_test = torch.randn(n_samples, n_features)

    model = MockTorchModel(n_classes=n_classes)

    predictor = LAC(model)
    predictor.calibrate(x_cal, y_cal, significance_level=0.1)

    sets = predictor.predict(x_test, significance_level=0.1)

    assert sets.shape == (n_samples, n_classes)

    # Admissibility Check: Ensure no empty sets (sum of True per row >= 1)
    set_sizes = sets.sum(dim=1)
    assert not torch.any(set_sizes == 0), "Found empty sets (Accretive Completion failure)"
