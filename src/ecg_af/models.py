from __future__ import annotations

import tensorflow as tf


def _default_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]


def build_1d_cnn(
    input_shape: tuple[int, int] = (9000, 1),
) -> tf.keras.Model:
    """Build the original baseline 1D CNN used for AF classification."""
    inputs = tf.keras.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=7,
        padding="same",
        activation="relu",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="af_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="baseline_1dcnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def build_regularised_1d_cnn(
    input_shape: tuple[int, int] = (9000, 1),
) -> tf.keras.Model:
    """Build the regularised 1D CNN with global average pooling."""
    l2_reg = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.Conv1D(
        32,
        kernel_size=15,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2_reg,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.SpatialDropout1D(0.10)(x)

    x = tf.keras.layers.Conv1D(
        64,
        kernel_size=11,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2_reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.SpatialDropout1D(0.15)(x)

    x = tf.keras.layers.Conv1D(
        128,
        kernel_size=7,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2_reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.SpatialDropout1D(0.20)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=l2_reg,
    )(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="af_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="regularised_1dcnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=_default_metrics(),
    )

    return model


def _cnn_feature_extractor(
    inputs: tf.Tensor,
) -> tf.Tensor:
    l2_reg = tf.keras.regularizers.l2(1e-4)

    x = inputs

    for filters, kernel_size, dropout in (
        (32, 15, 0.10),
        (64, 11, 0.15),
        (128, 7, 0.20),
    ):
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2_reg,
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
        x = tf.keras.layers.SpatialDropout1D(dropout)(x)

    return x


def build_cnn_lstm(
    input_shape: tuple[int, int] = (9000, 1),
) -> tf.keras.Model:
    """Build the CNN-LSTM model used in the dissertation experiments."""
    l2_reg = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=input_shape, name="ecg_input")
    x = _cnn_feature_extractor(inputs)

    x = tf.keras.layers.LSTM(
        64,
        dropout=0.20,
        recurrent_dropout=0.0,
        return_sequences=False,
        kernel_regularizer=l2_reg,
        name="lstm_layer",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=l2_reg,
    )(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="af_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=_default_metrics(),
    )

    return model


def build_cnn_gru(
    input_shape: tuple[int, int] = (9000, 1),
) -> tf.keras.Model:
    """Build the CNN-GRU model used in the dissertation experiments."""
    l2_reg = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=input_shape, name="ecg_input")
    x = _cnn_feature_extractor(inputs)

    x = tf.keras.layers.GRU(
        64,
        dropout=0.20,
        recurrent_dropout=0.0,
        return_sequences=False,
        kernel_regularizer=l2_reg,
        name="gru_layer",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=l2_reg,
    )(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="af_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn_gru",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=_default_metrics(),
    )

    return model


def residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    *,
    stride: int = 1,
    dropout_rate: float = 0.0,
    l2_value: float = 1e-4,
    block_name: str = "res_block",
) -> tf.Tensor:
    """Build one residual block for the 1D ResNet."""
    shortcut = x

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_value),
        name=f"{block_name}_conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name=f"{block_name}_bn1",
    )(x)
    x = tf.keras.layers.Activation(
        "relu",
        name=f"{block_name}_relu1",
    )(x)

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_value),
        name=f"{block_name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name=f"{block_name}_bn2",
    )(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2_value),
            name=f"{block_name}_shortcut_conv",
        )(shortcut)

        shortcut = tf.keras.layers.BatchNormalization(
            name=f"{block_name}_shortcut_bn",
        )(shortcut)

    x = tf.keras.layers.Add(
        name=f"{block_name}_add",
    )([x, shortcut])

    x = tf.keras.layers.Activation(
        "relu",
        name=f"{block_name}_output_relu",
    )(x)

    if dropout_rate > 0:
        x = tf.keras.layers.SpatialDropout1D(
            dropout_rate,
            name=f"{block_name}_dropout",
        )(x)

    return x


def build_resnet1d(
    input_shape: tuple[int, int] = (9000, 1),
) -> tf.keras.Model:
    """Build the 1D ResNet used for AF classification."""
    l2_value = 1e-4

    inputs = tf.keras.Input(shape=input_shape, name="ecg_input")

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=15,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_value),
        name="stem_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.Activation("relu", name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling1D(
        pool_size=4,
        strides=4,
        padding="same",
        name="stem_pool",
    )(x)

    x = residual_block(
        x,
        filters=32,
        kernel_size=9,
        dropout_rate=0.10,
        block_name="stage1_block1",
    )
    x = residual_block(
        x,
        filters=32,
        kernel_size=9,
        dropout_rate=0.10,
        block_name="stage1_block2",
    )

    x = residual_block(
        x,
        filters=64,
        kernel_size=7,
        stride=2,
        dropout_rate=0.15,
        block_name="stage2_block1",
    )
    x = residual_block(
        x,
        filters=64,
        kernel_size=7,
        dropout_rate=0.15,
        block_name="stage2_block2",
    )

    x = residual_block(
        x,
        filters=128,
        kernel_size=5,
        stride=2,
        dropout_rate=0.20,
        block_name="stage3_block1",
    )
    x = residual_block(
        x,
        filters=128,
        kernel_size=5,
        dropout_rate=0.20,
        block_name="stage3_block2",
    )

    x = tf.keras.layers.GlobalAveragePooling1D(
        name="global_average_pooling",
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value),
        name="dense_64",
    )(x)
    x = tf.keras.layers.Dropout(
        0.40,
        name="final_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="af_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="resnet1d",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=_default_metrics(),
    )

    return model