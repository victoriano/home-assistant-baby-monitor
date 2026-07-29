from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import TASKS
from .dataset import FrameExample, read_manifest
from .metrics import binary_metrics, select_threshold
from .model import build_multitask_model


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    height: int = 96
    width: int = 160
    batch_size: int = 64
    epochs: int = 24
    patience: int = 5
    learning_rate: float = 0.001
    seed: int = 20260723
    representative_samples: int = 256
    tensor_arena_kb: int = 768


def _image_path(frames_dir: Path, example: FrameExample) -> str:
    root = frames_dir.resolve()
    candidate = (root / example.relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"frame path escapes the frames directory: {example.relative_path}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return str(candidate)


def _class_weight_map(examples: tuple[FrameExample, ...], task: str, cap: float = 12.0) -> dict[int, float]:
    eligible = [example for example in examples if example.mask(task)]
    positives = sum(example.target(task) for example in eligible)
    negatives = len(eligible) - positives
    if not positives or not negatives:
        raise ValueError(f"{task} training data must contain positive and negative examples")
    raw = {
        0: min(cap, len(eligible) / (2 * negatives)),
        1: min(cap, len(eligible) / (2 * positives)),
    }
    total_weight = sum(raw[example.target(task)] for example in eligible)
    scale = len(examples) / total_weight
    return {target: weight * scale for target, weight in raw.items()}


def _weights_for(
    examples: tuple[FrameExample, ...],
    class_weights: dict[str, dict[int, float]],
) -> dict[str, np.ndarray]:
    return {
        task: np.asarray(
            [class_weights[task][example.target(task)] if example.mask(task) else 0.0 for example in examples],
            dtype=np.float32,
        ).reshape(-1, 1)
        for task in TASKS
    }


def _load_image(
    tensorflow: Any,
    path: Any,
    crop: Any,
    height: int,
    width: int,
    training: bool,
) -> Any:
    image = tensorflow.io.read_file(path)
    image = tensorflow.io.decode_image(image, channels=1, expand_animations=False)
    image.set_shape((None, None, 1))
    x0, y0, x1, y1 = tensorflow.unstack(crop)
    image = tensorflow.image.crop_and_resize(
        tensorflow.expand_dims(tensorflow.cast(image, tensorflow.float32), axis=0),
        boxes=tensorflow.reshape(tensorflow.stack((y0, x0, y1, x1)), (1, 4)),
        box_indices=tensorflow.zeros((1,), dtype=tensorflow.int32),
        crop_size=(height, width),
        method="bilinear",
    )
    image = image[0] / 255.0
    if training:
        image = tensorflow.image.random_flip_left_right(image)
        image = tensorflow.image.random_brightness(image, max_delta=0.06)
        image = tensorflow.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tensorflow.clip_by_value(image, 0.0, 1.0)
    return tensorflow.ensure_shape(image, (height, width, 1))


def _fit_dataset(
    tensorflow: Any,
    examples: tuple[FrameExample, ...],
    frames_dir: Path,
    config: TrainingConfig,
    class_weights: dict[str, dict[int, float]],
    *,
    training: bool,
) -> Any:
    paths = np.asarray([_image_path(frames_dir, example) for example in examples])
    crops = np.asarray(
        [(example.crop_x0, example.crop_y0, example.crop_x1, example.crop_y1) for example in examples],
        dtype=np.float32,
    )
    targets = {
        task: np.asarray([example.target(task) for example in examples], dtype=np.float32).reshape(-1, 1)
        for task in TASKS
    }
    weights = _weights_for(examples, class_weights)
    dataset = tensorflow.data.Dataset.from_tensor_slices((paths, crops, targets, weights))
    if training:
        dataset = dataset.shuffle(min(len(examples), 4096), seed=config.seed, reshuffle_each_iteration=True)

    def load(path: Any, crop: Any, labels: Any, sample_weights: Any) -> tuple[Any, Any, Any]:
        return (
            _load_image(tensorflow, path, crop, config.height, config.width, training),
            labels,
            sample_weights,
        )

    dataset = dataset.map(load, num_parallel_calls=tensorflow.data.AUTOTUNE)
    return dataset.batch(config.batch_size).prefetch(tensorflow.data.AUTOTUNE)


def _image_dataset(
    tensorflow: Any,
    examples: tuple[FrameExample, ...],
    frames_dir: Path,
    config: TrainingConfig,
    *,
    batch_size: int | None = None,
) -> Any:
    paths = np.asarray([_image_path(frames_dir, example) for example in examples])
    crops = np.asarray(
        [(example.crop_x0, example.crop_y0, example.crop_x1, example.crop_y1) for example in examples],
        dtype=np.float32,
    )
    dataset = tensorflow.data.Dataset.from_tensor_slices((paths, crops))
    dataset = dataset.map(
        lambda path, crop: _load_image(tensorflow, path, crop, config.height, config.width, False),
        num_parallel_calls=tensorflow.data.AUTOTUNE,
    )
    return dataset.batch(batch_size or config.batch_size).prefetch(tensorflow.data.AUTOTUNE)


def _predictions_by_task(model: Any, dataset: Any) -> dict[str, np.ndarray]:
    predictions = model.predict(dataset, verbose=0)
    if isinstance(predictions, dict):
        return {task: np.asarray(predictions[task]).reshape(-1) for task in TASKS}
    if isinstance(predictions, list) and len(predictions) == len(TASKS):
        return {
            task: np.asarray(prediction).reshape(-1)
            for task, prediction in zip(model.output_names, predictions, strict=True)
        }
    raise TypeError("unexpected Keras prediction structure")


def _runtime_view(
    examples: tuple[FrameExample, ...],
    predictions: dict[str, np.ndarray],
) -> tuple[tuple[FrameExample, ...], dict[str, np.ndarray]]:
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.frame_id, []).append(index)
    runtime_examples: list[FrameExample] = []
    runtime_predictions = {task: [] for task in TASKS}
    for indexes in groups.values():
        winner = max(indexes, key=lambda index: float(predictions["presence"][index]))
        labeled = next((index for index in indexes if examples[index].presence_target), indexes[0])
        runtime_examples.append(examples[labeled])
        runtime_predictions["presence"].append(max(float(predictions["presence"][index]) for index in indexes))
        for task in ("awake", "pacifier"):
            runtime_predictions[task].append(float(predictions[task][winner]))
    return tuple(runtime_examples), {
        task: np.asarray(values, dtype=np.float32) for task, values in runtime_predictions.items()
    }


def _evaluate(
    examples: tuple[FrameExample, ...],
    predictions: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    examples, predictions = _runtime_view(examples, predictions)
    report: dict[str, Any] = {"overall": {}, "locations": {}}

    def task_metrics(indexes: list[int], task: str) -> dict[str, Any]:
        eligible = [index for index in indexes if examples[index].mask(task)]
        truth = np.asarray([examples[index].target(task) for index in eligible])
        scores = np.asarray([predictions[task][index] for index in eligible])
        return binary_metrics(truth, scores, thresholds[task])

    all_indexes = list(range(len(examples)))
    for task in TASKS:
        report["overall"][task] = task_metrics(all_indexes, task)
    for location in sorted({example.location_id for example in examples}):
        indexes = [index for index, example in enumerate(examples) if example.location_id == location]
        report["locations"][location] = {task: task_metrics(indexes, task) for task in TASKS}
    return report


def _select_thresholds(
    examples: tuple[FrameExample, ...],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, Any]]:
    examples, predictions = _runtime_view(examples, predictions)
    thresholds: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    for task in TASKS:
        eligible = [index for index, example in enumerate(examples) if example.mask(task)]
        truth = np.asarray([examples[index].target(task) for index in eligible])
        scores = np.asarray([predictions[task][index] for index in eligible])
        threshold, task_metrics = select_threshold(truth, scores)
        thresholds[task] = threshold
        metrics[task] = task_metrics
    return thresholds, metrics


def _representative_examples(examples: tuple[FrameExample, ...], count: int) -> tuple[FrameExample, ...]:
    ordered = sorted(
        examples,
        key=lambda example: hashlib.sha256(f"representative:{example.frame_id}".encode()).hexdigest(),
    )
    return tuple(ordered[: min(count, len(ordered))])


def _convert_int8(
    tensorflow: Any,
    model: Any,
    representative_dataset: Any,
) -> bytes:
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]

    def representative() -> Any:
        for batch in representative_dataset:
            yield [batch.numpy()]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tensorflow.int8
    converter.inference_output_type = tensorflow.int8
    return converter.convert()


def _dequantize(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if not scale:
        return value.astype(np.float32)
    return (value.astype(np.float32) - float(zero_point)) * float(scale)


def _match_output_slots(
    tensorflow: Any,
    model_path: Path,
    examples: tuple[FrameExample, ...],
    frames_dir: Path,
    config: TrainingConfig,
    reference: dict[str, np.ndarray],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    interpreter = tensorflow.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    sample_count = min(48, len(examples))
    slot_values = [[] for _ in output_details]
    dataset = _image_dataset(
        tensorflow,
        examples[:sample_count],
        frames_dir,
        config,
        batch_size=1,
    )
    input_scale, input_zero = input_detail["quantization"]
    for batch in dataset:
        quantized = np.rint(batch.numpy() / input_scale + input_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        for slot, detail in enumerate(output_details):
            value = interpreter.get_tensor(detail["index"]).reshape(-1)
            slot_values[slot].append(float(_dequantize(value, detail)[0]))
    slot_arrays = [np.asarray(values) for values in slot_values]
    best: tuple[float, tuple[int, ...]] | None = None
    for permutation in itertools.permutations(range(len(output_details))):
        error = sum(
            float(np.mean(np.square(slot_arrays[permutation[index]] - reference[task][:sample_count])))
            for index, task in enumerate(TASKS)
        )
        if best is None or error < best[0]:
            best = (error, permutation)
    assert best is not None
    mapping = {task: best[1][index] for index, task in enumerate(TASKS)}
    return mapping, output_details


def _tflite_predictions(
    tensorflow: Any,
    model_path: Path,
    examples: tuple[FrameExample, ...],
    frames_dir: Path,
    config: TrainingConfig,
    output_slots: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    interpreter = tensorflow.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    input_scale, input_zero = input_detail["quantization"]
    values = {task: [] for task in TASKS}
    dataset = _image_dataset(tensorflow, examples, frames_dir, config, batch_size=1)
    started = time.monotonic()
    for batch in dataset:
        quantized = np.rint(batch.numpy() / input_scale + input_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized)
        interpreter.invoke()
        for task in TASKS:
            detail = output_details[output_slots[task]]
            raw = interpreter.get_tensor(detail["index"]).reshape(-1)
            values[task].append(float(_dequantize(raw, detail)[0]))
    elapsed = time.monotonic() - started
    timing = {
        "host_seconds": elapsed,
        "host_milliseconds_per_frame": elapsed * 1000 / max(1, len(examples)),
    }
    return {task: np.asarray(task_values) for task, task_values in values.items()}, timing


def _quantization(detail: dict[str, Any]) -> dict[str, Any]:
    scale, zero_point = detail["quantization"]
    return {
        "dtype": np.dtype(detail["dtype"]).name,
        "shape": [int(value) for value in detail["shape"]],
        "scale": float(scale),
        "zero_point": int(zero_point),
        "tensor_index": int(detail["index"]),
        "name": str(detail["name"]),
    }


def _write_c_model(model_bytes: bytes, generated_dir: Path) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    header = """#pragma once

extern const unsigned char g_baby_monitor_model[];
extern const unsigned int g_baby_monitor_model_len;
"""
    lines = []
    for index in range(0, len(model_bytes), 12):
        chunk = ", ".join(f"0x{value:02x}" for value in model_bytes[index : index + 12])
        lines.append(f"  {chunk},")
    source = (
        '#include "model_data.h"\n\n'
        "alignas(16) const unsigned char g_baby_monitor_model[] = {\n" + "\n".join(lines) + "\n};\n\n"
        f"const unsigned int g_baby_monitor_model_len = {len(model_bytes)};\n"
    )
    (generated_dir / "model_data.h").write_text(header, encoding="utf-8")
    (generated_dir / "model_data.cc").write_text(source, encoding="utf-8")


def _write_metadata_header(metadata: dict[str, Any], generated_dir: Path) -> None:
    input_meta = metadata["input"]
    outputs = metadata["outputs"]
    thresholds = metadata["thresholds"]
    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "namespace baby_monitor_model {",
        f"inline constexpr int kInputHeight = {input_meta['shape'][1]};",
        f"inline constexpr int kInputWidth = {input_meta['shape'][2]};",
        f"inline constexpr int kInputChannels = {input_meta['shape'][3]};",
        f"inline constexpr float kInputScale = {input_meta['scale']:.10g}f;",
        f"inline constexpr int kInputZeroPoint = {input_meta['zero_point']};",
        f"inline constexpr std::size_t kTensorArenaBytes = {metadata['tensor_arena_bytes']};",
    ]
    for task in TASKS:
        title = task.title()
        output = outputs[task]
        lines.extend(
            (
                f"inline constexpr int k{title}OutputSlot = {output['slot']};",
                f"inline constexpr float k{title}OutputScale = {output['scale']:.10g}f;",
                f"inline constexpr int k{title}OutputZeroPoint = {output['zero_point']};",
                f"inline constexpr float k{title}Threshold = {thresholds[task]:.10g}f;",
            )
        )
    lines.extend(("}  // namespace baby_monitor_model", ""))
    (generated_dir / "model_metadata.h").write_text("\n".join(lines), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Baby Monitor edge model report",
        "",
        f"- Model: full-integer int8, {report['model']['bytes']:,} bytes",
        f"- Input: {report['model']['input_width']}×{report['model']['input_height']} grayscale",
        "- Test split: newest capture days per location; no location/day appears in training.",
        "",
        "## Quantized test metrics",
        "",
        "| Task | Samples | Positive | Recall | Specificity | F1 | Balanced accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        metrics = report["quantized_test"]["overall"][task]

        def percent(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.1%}"

        lines.append(
            f"| {task} | {metrics['samples']} | {metrics['positive']} | "
            f"{percent(metrics['recall'])} | {percent(metrics['specificity'])} | "
            f"{percent(metrics['f1'])} | {percent(metrics['balanced_accuracy'])} |"
        )
    lines.extend(
        (
            "",
            "These metrics measure agreement with the existing AI labels, not clinical truth. "
            "Review the stratified queue and test on the physical camera before enabling side effects.",
            "",
        )
    )
    return "\n".join(lines)


def train_and_export(
    manifest_path: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    generated_dir: Path | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    import tensorflow as tf

    config = config or TrainingConfig()
    if config.height <= 0 or config.width <= 0 or config.batch_size <= 0 or config.epochs <= 0:
        raise ValueError("image dimensions, batch size, and epochs must be positive")
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()

    all_examples = read_manifest(manifest_path)
    by_split = {
        split: tuple(example for example in all_examples if example.split == split)
        for split in ("train", "validation", "test")
    }
    if any(not examples for examples in by_split.values()):
        raise ValueError("manifest must contain non-empty train, validation, and test splits")
    class_weights = {task: _class_weight_map(by_split["train"], task) for task in TASKS}
    train_dataset = _fit_dataset(
        tf,
        by_split["train"],
        frames_dir,
        config,
        class_weights,
        training=True,
    )
    validation_dataset = _fit_dataset(
        tf,
        by_split["validation"],
        frames_dir,
        config,
        class_weights,
        training=False,
    )

    model = build_multitask_model(tf, height=config.height, width=config.width)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={task: tf.keras.losses.BinaryCrossentropy() for task in TASKS},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=max(2, config.patience // 2),
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    model = tf.keras.models.load_model(best_model_path)
    (output_dir / "history.json").write_text(
        json.dumps(
            {key: [float(value) for value in values] for key, values in history.history.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    validation_images = _image_dataset(tf, by_split["validation"], frames_dir, config)
    validation_predictions = _predictions_by_task(model, validation_images)
    thresholds, validation_metrics = _select_thresholds(by_split["validation"], validation_predictions)
    test_images = _image_dataset(tf, by_split["test"], frames_dir, config)
    float_test_predictions = _predictions_by_task(model, test_images)
    float_test = _evaluate(by_split["test"], float_test_predictions, thresholds)

    representative_examples = _representative_examples(
        by_split["train"],
        config.representative_samples,
    )
    representative_dataset = _image_dataset(
        tf,
        representative_examples,
        frames_dir,
        config,
        batch_size=1,
    )
    model_bytes = _convert_int8(tf, model, representative_dataset)
    model_path = output_dir / "baby_monitor_multitask_int8.tflite"
    model_path.write_bytes(model_bytes)
    output_slots, output_details = _match_output_slots(
        tf,
        model_path,
        by_split["test"],
        frames_dir,
        config,
        float_test_predictions,
    )
    quantized_predictions, host_timing = _tflite_predictions(
        tf,
        model_path,
        by_split["test"],
        frames_dir,
        config,
        output_slots,
    )
    quantized_test = _evaluate(by_split["test"], quantized_predictions, thresholds)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    metadata = {
        "format": "TensorFlow Lite Micro full-integer int8",
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "bytes": len(model_bytes),
        "input": _quantization(input_detail),
        "outputs": {
            task: {
                **_quantization(output_details[output_slots[task]]),
                "slot": output_slots[task],
            }
            for task in TASKS
        },
        "thresholds": thresholds,
        "tensor_arena_bytes": config.tensor_arena_kb * 1024,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    generated = generated_dir or output_dir / "generated"
    _write_c_model(model_bytes, generated)
    _write_metadata_header(metadata, generated)
    report = {
        "config": asdict(config),
        "tensorflow_version": tf.__version__,
        "class_weights": class_weights,
        "validation_threshold_selection": validation_metrics,
        "float_test": float_test,
        "quantized_test": quantized_test,
        "host_tflite_timing": host_timing,
        "model": {
            "path": model_path.name,
            "bytes": len(model_bytes),
            "sha256": metadata["sha256"],
            "input_width": config.width,
            "input_height": config.height,
            "parameters": int(model.count_params()),
        },
        "dataset": {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            **{
                split: {
                    "samples": len(examples),
                    "frames": len({example.frame_id for example in examples}),
                }
                for split, examples in by_split.items()
            },
        },
        "roi_profiles": {
            location: sorted(
                {
                    (
                        example.crop_name,
                        example.crop_x0,
                        example.crop_y0,
                        example.crop_x1,
                        example.crop_y1,
                    )
                    for example in all_examples
                    if example.location_id == location
                }
            )
            for location in sorted({example.location_id for example in all_examples})
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report
