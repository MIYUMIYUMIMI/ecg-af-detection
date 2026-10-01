from __future__ import annotations

import numpy as np

from ecg_af.acge import (
    build_acge_model,
    build_gating_features,
    compute_balanced_class_weights,
)


def test_build_gating_features_shape() -> None:
    cnn = np.array([0.1, 0.8, 0.4], dtype=np.float32)
    gru = np.array([0.2, 0.7, 0.5], dtype=np.float32)
    resnet = np.array([0.3, 0.9, 0.6], dtype=np.float32)

    features = build_gating_features(
        cnn,
        gru,
        resnet,
    )

    assert features.shape == (3, 13)


def test_first_three_gating_features_are_base_probabilities() -> None:
    cnn = np.array([0.1, 0.8], dtype=np.float32)
    gru = np.array([0.2, 0.7], dtype=np.float32)
    resnet = np.array([0.3, 0.9], dtype=np.float32)

    features = build_gating_features(
        cnn,
        gru,
        resnet,
    )

    np.testing.assert_allclose(
        features[:, :3],
        np.column_stack(
            [
                cnn,
                gru,
                resnet,
            ]
        ),
    )


def test_acge_model_outputs_probability() -> None:
    model = build_acge_model()

    X = np.zeros(
        (2, 13),
        dtype=np.float32,
    )

    prediction = model.predict(
        X,
        verbose=0,
    )

    assert prediction.shape == (2, 1)
    assert np.all(prediction >= 0.0)
    assert np.all(prediction <= 1.0)


def test_balanced_class_weights_returns_two_classes() -> None:
    y = np.array(
        [0, 0, 0, 1],
        dtype=np.int32,
    )

    weights = compute_balanced_class_weights(y)

    assert set(weights) == {0, 1}
    assert weights[1] > weights[0]