from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> tuple[float, pd.DataFrame]:
    """Choose the probability threshold that maximizes AF-class F1."""
    thresholds = np.arange(
        start,
        stop + step / 2,
        step,
    )

    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(np.int32)

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(
                    accuracy_score(
                        y_true,
                        predictions,
                    )
                ),
                "af_precision": float(
                    precision_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "af_recall": float(
                    recall_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "af_f1": float(
                    f1_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
            }
        )

    threshold_df = pd.DataFrame(rows)

    best_index = threshold_df["af_f1"].idxmax()
    best_threshold = float(
        threshold_df.loc[
            best_index,
            "threshold",
        ]
    )

    return best_threshold, threshold_df


def evaluate_binary_classifier(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[
    dict[str, float | int],
    np.ndarray,
    np.ndarray,
    str,
]:
    """Evaluate AF-vs-non-AF probabilities at a fixed threshold."""
    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "af_precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "af_recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "af_f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "non_af_f1": float(
            f1_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    report = classification_report(
        y_true,
        predictions,
        target_names=["Non-AF", "AF"],
        digits=4,
        zero_division=0,
    )

    return metrics, matrix, predictions, report