from __future__ import annotations

from typing import Any

from . import TASKS


def _depthwise_block(
    keras: Any,
    value: Any,
    filters: int,
    stride: int,
    name: str,
) -> Any:
    value = keras.layers.DepthwiseConv2D(
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_depthwise",
    )(value)
    value = keras.layers.BatchNormalization(name=f"{name}_depthwise_bn")(value)
    value = keras.layers.ReLU(max_value=6, name=f"{name}_depthwise_relu")(value)
    value = keras.layers.Conv2D(
        filters,
        1,
        padding="same",
        use_bias=False,
        name=f"{name}_pointwise",
    )(value)
    value = keras.layers.BatchNormalization(name=f"{name}_pointwise_bn")(value)
    return keras.layers.ReLU(max_value=6, name=f"{name}_pointwise_relu")(value)


def build_multitask_model(
    tensorflow: Any,
    *,
    height: int,
    width: int,
    dropout: float = 0.15,
) -> Any:
    """Build a small operator-constrained CNN suitable for TFLite Micro."""

    keras = tensorflow.keras
    inputs = keras.Input(shape=(height, width, 1), dtype="float32", name="image")
    value = keras.layers.Conv2D(
        16,
        3,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    value = keras.layers.BatchNormalization(name="stem_bn")(value)
    value = keras.layers.ReLU(max_value=6, name="stem_relu")(value)
    blocks = (
        (24, 1),
        (24, 1),
        (32, 2),
        (32, 1),
        (48, 2),
        (48, 1),
        (64, 2),
        (96, 1),
        (128, 2),
    )
    for index, (filters, stride) in enumerate(blocks, start=1):
        value = _depthwise_block(keras, value, filters, stride, f"block_{index}")
    value = keras.layers.GlobalAveragePooling2D(name="global_average")(value)
    value = keras.layers.Dropout(dropout, name="dropout")(value)
    outputs = {task: keras.layers.Dense(1, activation="sigmoid", name=task)(value) for task in TASKS}
    return keras.Model(inputs=inputs, outputs=outputs, name="baby_monitor_edge")
