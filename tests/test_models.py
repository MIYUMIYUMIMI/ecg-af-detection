from __future__ import annotations

import numpy as np

from ecg_af.models import (
    build_1d_cnn,
    build_cnn_gru,
    build_cnn_lstm,
    build_regularised_1d_cnn,
    build_resnet1d,
)


def _assert_binary_model_output(model) -> None:
    X = np.zeros(
        (2, 9000, 1),
        dtype=np.float32,
    )

    prediction = model.predict(
        X,
        verbose=0,
    )

    assert prediction.shape == (2, 1)
    assert np.all(prediction >= 0.0)
    assert np.all(prediction <= 1.0)


def test_baseline_1d_cnn_output_shape() -> None:
    model = build_1d_cnn()
    _assert_binary_model_output(model)


def test_regularised_1d_cnn_output_shape() -> None:
    model = build_regularised_1d_cnn()
    _assert_binary_model_output(model)


def test_cnn_lstm_output_shape() -> None:
    model = build_cnn_lstm()
    _assert_binary_model_output(model)


def test_cnn_gru_output_shape() -> None:
    model = build_cnn_gru()
    _assert_binary_model_output(model)


def test_resnet1d_output_shape() -> None:
    model = build_resnet1d()
    _assert_binary_model_output(model)