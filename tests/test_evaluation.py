from __future__ import annotations

import numpy as np

from ecg_af.evaluation import (
    evaluate_binary_classifier,
    tune_threshold,
)


def test_tune_threshold_finds_perfect_split() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9],
        dtype=np.float32,
    )

    threshold, results = tune_threshold(
        y_true,
        probabilities,
        start=0.1,
        stop=0.9,
        step=0.1,
    )

    best_row = results.loc[
        results["threshold"] == threshold
    ].iloc[0]

    assert best_row["af_f1"] == 1.0


def test_evaluate_binary_classifier_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    probabilities = np.array(
        [0.1, 0.2, 0.8, 0.9],
        dtype=np.float32,
    )

    metrics, matrix, predictions, report = (
        evaluate_binary_classifier(
            y_true,
            probabilities,
            threshold=0.5,
        )
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["af_precision"] == 1.0
    assert metrics["af_recall"] == 1.0
    assert metrics["af_f1"] == 1.0
    assert metrics["specificity"] == 1.0

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2

    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [2, 0],
                [0, 2],
            ]
        ),
    )

    np.testing.assert_array_equal(
        predictions,
        np.array([0, 0, 1, 1]),
    )

    assert "Non-AF" in report
    assert "AF" in report


def test_specificity_calculation() -> None:
    y_true = np.array([0, 0, 0, 1], dtype=np.int32)
    probabilities = np.array(
        [0.1, 0.7, 0.2, 0.9],
        dtype=np.float32,
    )

    metrics, _, _, _ = evaluate_binary_classifier(
        y_true,
        probabilities,
        threshold=0.5,
    )

    # TN = 2, FP = 1
    assert np.isclose(
        metrics["specificity"],
        2 / 3,
    )