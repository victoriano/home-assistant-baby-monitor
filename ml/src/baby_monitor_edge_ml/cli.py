from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import prepare_dataset
from .training import TrainingConfig, train_and_export
from .yolo_training import (
    PoseHeadConfig,
    YoloTrainingConfig,
    evaluate_yolo_classifiers,
    prepare_yolo_dataset,
    train_yolo_classifiers,
)


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_dataset(
        args.database,
        args.frames_dir,
        args.output_dir,
        min_confidence=args.min_confidence,
        overrides_path=args.overrides,
        require_visible_face_for_pacifier=not args.allow_hidden_face_for_pacifier,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        roi_config_path=args.roi_config,
        capture_windows_path=args.capture_windows,
    )


def _train(args: argparse.Namespace) -> dict[str, Any]:
    config = TrainingConfig(
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        representative_samples=args.representative_samples,
        tensor_arena_kb=args.tensor_arena_kb,
    )
    return train_and_export(
        args.manifest,
        args.frames_dir,
        args.output_dir,
        generated_dir=args.generated_dir,
        config=config,
    )


def _prepare_yolo(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_yolo_dataset(
        args.manifest,
        args.frames_dir,
        args.output_dir,
        image_size=args.image_size,
        seed=args.seed,
        max_minority_repeats=args.max_minority_repeats,
        temporal_consensus=not args.keep_isolated_detail_labels,
        pose_model_path=args.pose_model,
        pose_config=PoseHeadConfig(
            image_size=args.pose_image_size,
            batch_size=args.pose_batch_size,
            device=args.pose_device,
            detection_confidence=args.pose_detection_confidence,
            nose_confidence=args.pose_nose_confidence,
            head_keypoint_confidence=args.pose_head_keypoint_confidence,
        ),
        overwrite=args.overwrite,
    )


def _train_yolo(args: argparse.Namespace) -> dict[str, Any]:
    return train_yolo_classifiers(
        args.dataset_dir,
        args.output_dir,
        config=YoloTrainingConfig(
            base_model=args.base_model,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            workers=args.workers,
        ),
        overwrite=args.overwrite,
        resume=args.resume,
    )


def _evaluate_yolo(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_yolo_classifiers(
        args.dataset_dir,
        args.artifact_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baby-monitor-edge",
        description="Build and train a private TinyML vision model from Baby Monitor history.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a private manifest and chronological data split.")
    prepare.add_argument("--database", type=Path, required=True)
    prepare.add_argument("--frames-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--overrides", type=Path)
    prepare.add_argument("--min-confidence", type=float, default=0.8)
    prepare.add_argument("--validation-fraction", type=float, default=0.15)
    prepare.add_argument("--test-fraction", type=float, default=0.15)
    prepare.add_argument("--allow-hidden-face-for-pacifier", action="store_true")
    prepare.add_argument("--roi-config", type=Path)
    prepare.add_argument("--capture-windows", type=Path)
    prepare.set_defaults(handler=_prepare)

    train = commands.add_parser("train", help="Train, quantify, test, and export a TFLite Micro model.")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--frames-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--generated-dir", type=Path)
    train.add_argument("--height", type=int, default=96)
    train.add_argument("--width", type=int, default=160)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=24)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=20260723)
    train.add_argument("--representative-samples", type=int, default=256)
    train.add_argument("--tensor-arena-kb", type=int, default=768)
    train.set_defaults(handler=_train)

    prepare_yolo = commands.add_parser(
        "prepare-yolo",
        help="Materialize balanced YOLO classification datasets from an existing private manifest.",
    )
    prepare_yolo.add_argument("--manifest", type=Path, required=True)
    prepare_yolo.add_argument("--frames-dir", type=Path, required=True)
    prepare_yolo.add_argument("--output-dir", type=Path, required=True)
    prepare_yolo.add_argument("--image-size", type=int, default=320)
    prepare_yolo.add_argument("--seed", type=int, default=20260729)
    prepare_yolo.add_argument("--max-minority-repeats", type=int, default=6)
    prepare_yolo.add_argument("--keep-isolated-detail-labels", action="store_true")
    prepare_yolo.add_argument(
        "--pose-model",
        type=Path,
        help="Optional YOLO pose weights used to crop the baby's head for detail tasks.",
    )
    prepare_yolo.add_argument("--pose-image-size", type=int, default=640)
    prepare_yolo.add_argument("--pose-batch-size", type=int, default=16)
    prepare_yolo.add_argument("--pose-device", default="mps")
    prepare_yolo.add_argument("--pose-detection-confidence", type=float, default=0.2)
    prepare_yolo.add_argument("--pose-nose-confidence", type=float, default=0.45)
    prepare_yolo.add_argument("--pose-head-keypoint-confidence", type=float, default=0.3)
    prepare_yolo.add_argument("--overwrite", action="store_true")
    prepare_yolo.set_defaults(handler=_prepare_yolo)

    train_yolo = commands.add_parser(
        "train-yolo",
        help="Fine-tune private YOLO26 classifiers for presence, awake state, and pacifier.",
    )
    train_yolo.add_argument("--dataset-dir", type=Path, required=True)
    train_yolo.add_argument("--output-dir", type=Path, required=True)
    train_yolo.add_argument("--base-model", default="yolo26n-cls.pt")
    train_yolo.add_argument("--image-size", type=int, default=320)
    train_yolo.add_argument("--batch-size", type=int, default=8)
    train_yolo.add_argument("--epochs", type=int, default=40)
    train_yolo.add_argument("--patience", type=int, default=8)
    train_yolo.add_argument("--learning-rate", type=float, default=0.001)
    train_yolo.add_argument("--device", default="mps")
    train_yolo.add_argument("--seed", type=int, default=20260729)
    train_yolo.add_argument("--workers", type=int, default=0)
    train_yolo.add_argument("--overwrite", action="store_true")
    train_yolo.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed task models in a marked interrupted artifact directory.",
    )
    train_yolo.set_defaults(handler=_train_yolo)

    evaluate_yolo = commands.add_parser(
        "evaluate-yolo",
        help="Calibrate validation thresholds and report untouched YOLO test metrics.",
    )
    evaluate_yolo.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_yolo.add_argument("--artifact-dir", type=Path, required=True)
    evaluate_yolo.add_argument("--device", default="cpu")
    evaluate_yolo.add_argument("--batch-size", type=int, default=4)
    evaluate_yolo.set_defaults(handler=_evaluate_yolo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
