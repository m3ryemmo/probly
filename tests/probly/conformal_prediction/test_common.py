"""Tests for common utilities and methods in conformal prediction."""

import numpy as np
import pytest
from probly.conformal_prediction.common import ConformalPredictor 

class DummyModel: 
    """Simple dummy model used only for testing.
    
    Its predict() method returns the input as a numpy array.
    """

    def predict(self, x):
        return np.array(x, dtype=float)

class DummyConformalPredictor(ConformalPredictor):
    """Concrete test subclass of the abstract ConformalPredictor."""

    def compute_nonconformity(self, x, y):
        """Compute nonconformity scores as absolute differences."""
       
        if self.conformity_func is not None: 
           # Use the user-provided nonconformity function
            return self.conformity_func(x, y, self.model)

        # Default: absolute error 
        preds = self.model.predict(x)
        y = np.array(y, dtype=float)
        return np.abs(preds - y)
    
    def predict(self, x, significance_level):
        """Return prediction intervals around the model prediction.
        
        The method: 
        - requeries that the predictor is calibrated,
        - uses the stored threshold as a symmetric radius,
        - return a list of (low, high) tuples
        """

        if not self.is_calibrated or self.threshold is None:
            raise RuntimeError("Predictor is not calibrated")
        
        preds = self.model.predict(x)

        intervals = [] 
        for p in preds: 
            low = float(p - self.threshold)
            high = float(p + self.threshold) 
            intervals.append((low, high))

        return intervals 

    
def test_initial_state_of_conformal_predictor():
    """Test that the ConformalPredictor starts with correcct default values."""

    model = DummyModel()
    cp = DummyConformalPredictor(model)

    # check that the model is sorted correctly 
    assert cp.model is model 

    # No nonconformity function by default 
    assert cp.conformity_func is None 

    # Calibration-related fields should be empty before calibration 
    assert cp.nonconformity_scores is None 
    assert cp.threshold is None 
    assert cp.is_calibrated is False

def test_calibrate_sets_scores_threshold_and_flag(): 
    """Calibration should compute scores, and mark the predictor as calibrated."""

    model = DummyModel()
    cp = DummyConformalPredictor(model)

    x_cal = [1, 2, 3, 4, 5]
    y_cal = [1, 2, 4, 4, 10] 
    alpha = 0.2 

    returned_threshold = cp.calibrate(x_cal, y_cal, alpha) 

    # Scores habe been computed 
    assert cp.nonconformity_scores is not None 
    assert len(cp.nonconformity_scores) == len(x_cal)

    # Threshold is correct 
    expected_threshold = float(np.quantile(cp.nonconformity_scores, 1 - alpha))
    assert returned_threshold == expected_threshold 
    assert cp.threshold == expected_threshold 

    # Predictor flagged as calibrated 
    assert cp.is_calibrated is True 

def test_predict_raises_error_if_not_calibrated():
    """predict() should raise an error if called before calibration."""

    model = DummyModel()
    cp = DummyConformalPredictor(model) 

    with pytest.raises(RuntimeError): 
        cp.predict([1, 2, 3], significance_level=0.1)

def test_predict_uses_threshold_after_calibration():
    """Prediction intervals should use the stored threshold after calibration."""

    model = DummyModel()
    cp = DummyConformalPredictor(model)

    # Calibration step
    x_cal = [1, 2, 3]
    y_cal = [1, 3, 5]
    alpha = 0.1
    cp.calibrate(x_cal, y_cal, alpha)

    # Prediction step 
    x_test = [10, 20]
    intervals = cp.predict(x_test, significance_level=alpha)

    # Number of intervals should match number of test points 
    assert len(intervals) == len(x_test) 

    # Each interval should be [x - threshold, x + threshold] 
    for (low, high), x in zip(intervals, x_test):
        assert pytest.approx(low) == x - cp.threshold
        assert pytest.approx(high) == x + cp.threshold

def test_custom_nonconformity_function_is_used(): 
    """A custom nonconformity function should override the default behaviour."""

    model = DummyModel()

    def custom_nc(x, y, model_):
        """Return constant nonconformity scores, independent of model and labels."""
        return np.ones(len(x), dtype=float) * 3.0
    
    cp = DummyConformalPredictor(model, nonconformity_func=custom_nc)

    x_cal = [0, 1, 2]
    y_cal = [0, 0, 0] 
    alpha = 0.2

    cp.calibrate(x_cal, y_cal, alpha) 

    # All stored scores should be exactly 3.0
    assert cp.nonconformity_scores is not None 
    assert np.all(cp.nonconformity_scores == 3.0)

    # Threshold should also be 3.0 
    assert cp.threshold is not None
    assert cp.threshold == 3.0

