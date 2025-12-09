"""Tests for Accretive completion LAC method."""

from __future__ import annotations

import numpy as np

from probly.conformal_prediction.lac.methods.accretive_completion import accretive_completion


def test_fills_empty_set_with_max_score() -> None:
    """Scenario: A prediction set is completely empty (all False).

    Expectation: The method must force the class with the highest score (0.8 -> Index 1) to True.
    """
    # Input: 1 Sample, 2 Classes. Both False.
    pred_set = np.array([[False, False]])
    # Scores: Class 0 = 0.2, Class 1 = 0.8
    scores = np.array([[0.2, 0.8]])

    result = accretive_completion(pred_set, scores)

    # 1. Verify that the set is no longer empty
    assert np.any(result), "The set should not be empty anymore."

    # 2. Verify that exactly [False, True] was selected
    expected = np.array([[False, True]])
    np.testing.assert_array_equal(result, expected, err_msg="The class with the highest score was not selected.")


def test_leaves_non_empty_set_untouched() -> None:
    """Scenario: A set is already filled/valid (contains at least one True).

    Expectation: The method must NOT modify the set, even if another class has a higher score.
    """
    # Input: Class 0 is True.
    pred_set = np.array([[True, False]])
    # Scores: Class 1 has a higher score (0.9),
    # but since the set is not empty, it should remain unchanged.
    scores = np.array([[0.1, 0.9]])

    result = accretive_completion(pred_set, scores)

    np.testing.assert_array_equal(result, pred_set, err_msg="A valid (non-empty) set must not be modified.")


def test_mixed_batch() -> None:
    """Scenario: Processing a batch with mixed cases (one empty, one valid).

    Row 0: Empty -> Must be repaired.
    Row 1: Full -> Must remain unchanged.
    """
    pred_sets = np.array(
        [
            [False, False],  # Empty -> needs repair
            [True, False],  # Full -> should stay as is
        ],
    )
    scores = np.array(
        [
            [0.7, 0.3],  # Row 0: Index 0 has highest score
            [0.4, 0.6],  # Row 1: Scores don't matter here
        ],
    )

    expected = np.array(
        [
            [True, False],  # Repaired: Index 0 forced to True
            [True, False],  # Unchanged
        ],
    )

    result = accretive_completion(pred_sets, scores)
    np.testing.assert_array_equal(result, expected)


def test_randomized_large_batch() -> None:
    """Stress-Test: Generates random data to simulate a 'real' dataset scenario.

    Checks if the logic holds for 100 samples and 10 classes with random values.
    """
    np.random.Generator(42)  # For reproducibility
    n_samples = 100
    n_classes = 10

    # 1. Random Scores (Probabilities between 0 and 1)
    scores = np.random.Generator(n_samples, n_classes)

    # 2. Random Prediction Sets (Boolean)
    # We generate sets where about 30% of rows might be empty (False)
    pred_sets = np.random.Generator([True, False], size=(n_samples, n_classes), p=[0.3, 0.7])

    # We manually force the first 10 rows to be empty to GUARANTEE we have work to do
    pred_sets[0:10, :] = False

    # Keep a copy of the original sets for comparison
    original_sets = pred_sets.copy()

    # Execute the function
    result = accretive_completion(pred_sets, scores)

    # CHECK 1: Every row must now have at least one 'True'
    row_sums = np.sum(result, axis=1)
    assert np.all(row_sums > 0), "Found empty sets after completion! The method failed to fix them."

    # CHECK 2: Detailed verification for every row
    for i in range(n_samples):
        original_row = original_sets[i]
        new_row = result[i]

        if np.sum(original_row) == 0:
            # Case A: Row was empty -> Must now include the class with the max score
            expected_idx = np.argmax(scores[i])
            assert new_row[expected_idx] == new_row[expected_idx], (
                f"Row {i}: Did not pick the class with the max score."
            )

            # In this simple setup, only exactly 1 element should have been added
            assert np.sum(new_row) == 1, f"Row {i}: Should have exactly 1 True value (the completed one)."
        else:
            # Case B: Row was already valid -> Must NOT have changed
            np.testing.assert_array_equal(
                new_row,
                original_row,
                err_msg=f"Row {i}: Modified a set that was already valid!",
            )
