from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.utils.class_weight import compute_class_weight


def build_gating_features(
    cnn_prob: np.ndarray,
    gru_prob: np.ndarray,
    resnet_prob: np.ndarray,
) -> np.ndarray:
    """Build the 13-dimensional feature vector used by ACGE.

    Features include:
    - three base-model probabilities,
    - three confidence values,
    - three pairwise disagreement values,
    - four probability summary statistics.
    """
    probabilities = np.column_stack(
        [
            cnn_prob,
            gru_prob,
            resnet_prob,
        ]
    ).astype(np.float32)

    p_cnn = probabilities[:, 0]
    p_gru = probabilities[:, 1]
    p_resnet = probabilities[:, 2]

    # Distance from 0.5 acts as a simple confidence measure.
    confidence = np.abs(
        probabilities - 0.5
    ).astype(np.float32)

    # Pairwise disagreement between base models.
    disagreement = np.column_stack(
        [
            np.abs(p_cnn - p_gru),
            np.abs(p_cnn - p_resnet),
            np.abs(p_gru - p_resnet),
        ]
    ).astype(np.float32)

    # Ensemble-level summary statistics.
    summary_features = np.column_stack(
        [
            probabilities.mean(axis=1),
            probabilities.std(axis=1),
            probabilities.max(axis=1),
            probabilities.min(axis=1),
        ]
    ).astype(np.float32)

    return np.column_stack(
        [
            probabilities,
            confidence,
            disagreement,
            summary_features,
        ]
    ).astype(np.float32)


def build_acge_model(
    input_dim: int = 13,
    hidden_units_1: int = 16,
    hidden_units_2: int = 8,
    dropout_rate: float = 0.20,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build the Adaptive Confidence-Guided Ensemble model."""
    feature_input = tf.keras.Input(
        shape=(input_dim,),
        name="gating_features",
    )

    x = tf.keras.layers.Dense(
        hidden_units_1,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="gating_dense_1",
    )(feature_input)

    x = tf.keras.layers.Dropout(
        dropout_rate,
        name="gating_dropout",
    )(x)

    x = tf.keras.layers.Dense(
        hidden_units_2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="gating_dense_2",
    )(x)

    dynamic_weights = tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="dynamic_model_weights",
    )(x)

    base_probabilities = tf.keras.layers.Lambda(
        lambda tensor: tensor[:, :3],
        name="base_probabilities",
    )(feature_input)

    final_probability = tf.keras.layers.Lambda(
        lambda tensors: tf.reduce_sum(
            tensors[0] * tensors[1],
            axis=1,
            keepdims=True,
        ),
        name="adaptive_fused_probability",
    )(
        [
            dynamic_weights,
            base_probabilities,
        ]
    )

    model = tf.keras.Model(
        inputs=feature_input,
        outputs=final_probability,
        name="ACGE",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),
            tf.keras.metrics.AUC(
                name="auc",
            ),
            tf.keras.metrics.Precision(
                name="precision",
            ),
            tf.keras.metrics.Recall(
                name="recall",
            ),
        ],
    )

    return model


def compute_balanced_class_weights(
    y: np.ndarray,
) -> dict[int, float]:
    """Compute balanced class weights for binary AF classification."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y,
    )

    return {
        0: float(weights[0]),
        1: float(weights[1]),
    }


def generate_acge_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
    epochs: int = 150,
    batch_size: int = 32,
    class_weight: dict[int, float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate out-of-fold ACGE predictions.

    Each outer fold is held out for OOF prediction. The remaining data are
    split again into training and early-stopping validation subsets.
    """
    if class_weight is None:
        class_weight = compute_balanced_class_weights(y)

    oof_prob = np.zeros(
        len(y),
        dtype=np.float32,
    )

    history_rows: list[dict[str, float | int]] = []

    outer_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for fold, (dev_idx, oof_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_dev = X[dev_idx]
        y_dev = y[dev_idx]

        X_oof = X[oof_idx]
        y_oof = y[oof_idx]

        inner_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=seed + fold,
        )

        inner_train_idx, inner_val_idx = next(
            inner_split.split(
                X_dev,
                y_dev,
            )
        )

        X_inner_train = X_dev[
            inner_train_idx
        ]
        y_inner_train = y_dev[
            inner_train_idx
        ]

        X_inner_val = X_dev[
            inner_val_idx
        ]
        y_inner_val = y_dev[
            inner_val_idx
        ]

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(
            seed + fold
        )

        fold_model = build_acge_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=15,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=6,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        history = fold_model.fit(
            X_inner_train,
            y_inner_train,
            validation_data=(
                X_inner_val,
                y_inner_val,
            ),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        fold_prob = fold_model.predict(
            X_oof,
            batch_size=batch_size,
            verbose=0,
        ).reshape(-1)

        oof_prob[oof_idx] = fold_prob

        best_epoch = int(
            np.argmin(
                history.history["val_loss"]
            )
            + 1
        )

        fold_auc = float(
            roc_auc_score(
                y_oof,
                fold_prob,
            )
        )

        history_rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_val_loss": float(
                    np.min(
                        history.history["val_loss"]
                    )
                ),
                "oof_auc": fold_auc,
                "oof_samples": len(oof_idx),
                "oof_af_samples": int(
                    y_oof.sum()
                ),
            }
        )

        del fold_model
        tf.keras.backend.clear_session()
        gc.collect()

    return (
        oof_prob,
        pd.DataFrame(history_rows),
    )


def select_final_epoch_count(
    fold_summary: pd.DataFrame,
    *,
    minimum_epochs: int = 10,
) -> int:
    """Select the final training length from OOF early-stopping results."""
    median_epoch = int(
        np.median(
            fold_summary["best_epoch"]
        )
    )

    return max(
        median_epoch,
        minimum_epochs,
    )


def train_final_acge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    epochs: int,
    batch_size: int = 32,
    class_weight: dict[int, float] | None = None,
) -> tuple[
    tf.keras.Model,
    tf.keras.callbacks.History,
]:
    """Train the final ACGE model on the full validation feature set."""
    if class_weight is None:
        class_weight = compute_balanced_class_weights(y)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_acge_model()

    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=0,
    )

    return model, history