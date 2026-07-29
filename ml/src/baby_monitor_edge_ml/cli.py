from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adult_training import (
    assemble_reviewed_adult_artifact,
    evaluate_adult_classifier,
    prepare_adult_dataset,
    train_adult_classifier,
)
from .dataset import prepare_dataset
from .detail_training import (
    assemble_reviewed_detail_artifact,
    evaluate_detail_classifiers,
    prepare_detail_dataset,
    rebalance_detail_validation,
    train_detail_classifiers,
)
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


def _prepare_yolo_details(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_detail_dataset(
        args.source_manifest,
        args.database,
        args.frames_dir,
        args.pose_model,
        args.output_dir,
        image_size=args.image_size,
        pose_config=PoseHeadConfig(
            image_size=args.pose_image_size,
            batch_size=args.pose_batch_size,
            device=args.pose_device,
            detection_confidence=args.pose_detection_confidence,
            nose_confidence=args.pose_nose_confidence,
            head_keypoint_confidence=args.pose_head_keypoint_confidence,
        ),
        seed=args.seed,
        max_minority_repeats=args.max_minority_repeats,
        temporal_consensus=not args.keep_isolated_detail_labels,
        overwrite=args.overwrite,
    )


def _train_yolo_details(args: argparse.Namespace) -> dict[str, Any]:
    return train_detail_classifiers(
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


def _rebalance_yolo_detail_validation(args: argparse.Namespace) -> dict[str, Any]:
    return rebalance_detail_validation(
        args.dataset_dir,
        seed=args.seed,
        max_minority_repeats=args.max_minority_repeats,
    )


def _evaluate_yolo_details(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_detail_classifiers(
        args.dataset_dir,
        args.artifact_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


def _assemble_yolo_details(args: argparse.Namespace) -> dict[str, Any]:
    return assemble_reviewed_detail_artifact(
        args.base_artifact,
        args.detail_artifact,
        args.output_dir,
        reviewed_tasks=tuple(args.reviewed_task),
        overwrite=args.overwrite,
    )


def _prepare_yolo_adults(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_adult_dataset(
        args.source_manifest,
        args.database,
        args.frames_dir,
        args.output_dir,
        image_size=args.image_size,
        seed=args.seed,
        max_minority_repeats=args.max_minority_repeats,
        temporal_consensus=not args.keep_isolated_labels,
        overwrite=args.overwrite,
    )


def _train_yolo_adults(args: argparse.Namespace) -> dict[str, Any]:
    return train_adult_classifier(
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
    )


def _evaluate_yolo_adults(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_adult_classifier(
        args.dataset_dir,
        args.artifact_dir,
        device=args.device,
        batch_size=args.batch_size,
    )


def _assemble_yolo_adults(args: argparse.Namespace) -> dict[str, Any]:
    return assemble_reviewed_adult_artifact(
        args.base_artifact,
        args.adult_artifact,
        args.output_dir,
        overwrite=args.overwrite,
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

    prepare_details = commands.add_parser(
        "prepare-yolo-details",
        help="Build private pose-localized head, body, and mouth classification datasets.",
    )
    prepare_details.add_argument("--source-manifest", type=Path, required=True)
    prepare_details.add_argument("--database", type=Path, required=True)
    prepare_details.add_argument("--frames-dir", type=Path, required=True)
    prepare_details.add_argument("--pose-model", type=Path, required=True)
    prepare_details.add_argument("--output-dir", type=Path, required=True)
    prepare_details.add_argument("--image-size", type=int, default=320)
    prepare_details.add_argument("--seed", type=int, default=20260730)
    prepare_details.add_argument("--max-minority-repeats", type=int, default=8)
    prepare_details.add_argument("--keep-isolated-detail-labels", action="store_true")
    prepare_details.add_argument("--pose-image-size", type=int, default=640)
    prepare_details.add_argument("--pose-batch-size", type=int, default=16)
    prepare_details.add_argument(
        "--pose-device",
        default="cpu",
        help="Pose preprocessing device. CPU is the stable default; MPS can stall in batched top-k.",
    )
    prepare_details.add_argument("--pose-detection-confidence", type=float, default=0.2)
    prepare_details.add_argument("--pose-nose-confidence", type=float, default=0.45)
    prepare_details.add_argument("--pose-head-keypoint-confidence", type=float, default=0.3)
    prepare_details.add_argument("--overwrite", action="store_true")
    prepare_details.set_defaults(handler=_prepare_yolo_details)

    train_details = commands.add_parser(
        "train-yolo-details",
        help="Train compact head-side, body-position, and mouth-open classifiers.",
    )
    train_details.add_argument("--dataset-dir", type=Path, required=True)
    train_details.add_argument("--output-dir", type=Path, required=True)
    train_details.add_argument("--base-model", default="yolo26n-cls.pt")
    train_details.add_argument("--image-size", type=int, default=320)
    train_details.add_argument("--batch-size", type=int, default=8)
    train_details.add_argument("--epochs", type=int, default=40)
    train_details.add_argument("--patience", type=int, default=8)
    train_details.add_argument("--learning-rate", type=float, default=0.001)
    train_details.add_argument("--device", default="mps")
    train_details.add_argument("--seed", type=int, default=20260730)
    train_details.add_argument("--workers", type=int, default=0)
    train_details.add_argument("--overwrite", action="store_true")
    train_details.add_argument("--resume", action="store_true")
    train_details.set_defaults(handler=_train_yolo_details)

    rebalance_details = commands.add_parser(
        "rebalance-yolo-detail-validation",
        help=(
            "Balance the private model-selection folders while preserving the "
            "natural validation rows used for calibration."
        ),
    )
    rebalance_details.add_argument("--dataset-dir", type=Path, required=True)
    rebalance_details.add_argument("--seed", type=int, default=20260730)
    rebalance_details.add_argument("--max-minority-repeats", type=int, default=8)
    rebalance_details.set_defaults(handler=_rebalance_yolo_detail_validation)

    evaluate_details = commands.add_parser(
        "evaluate-yolo-details",
        help="Calibrate abstention and score grouped secondary-feature holdouts.",
    )
    evaluate_details.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_details.add_argument("--artifact-dir", type=Path, required=True)
    evaluate_details.add_argument("--device", default="cpu")
    evaluate_details.add_argument("--batch-size", type=int, default=8)
    evaluate_details.set_defaults(handler=_evaluate_yolo_details)

    assemble_details = commands.add_parser(
        "assemble-yolo-details",
        help="Package explicitly reviewed, gate-passing secondary classifiers.",
    )
    assemble_details.add_argument("--base-artifact", type=Path, required=True)
    assemble_details.add_argument("--detail-artifact", type=Path, required=True)
    assemble_details.add_argument("--output-dir", type=Path, required=True)
    assemble_details.add_argument(
        "--reviewed-task",
        action="append",
        required=True,
        choices=("head_side", "body_position", "mouth_open"),
    )
    assemble_details.add_argument("--overwrite", action="store_true")
    assemble_details.set_defaults(handler=_assemble_yolo_details)

    prepare_adults = commands.add_parser(
        "prepare-yolo-adults",
        help="Build a private, grouped full-scene adult-presence dataset.",
    )
    prepare_adults.add_argument("--source-manifest", type=Path, required=True)
    prepare_adults.add_argument("--database", type=Path, required=True)
    prepare_adults.add_argument("--frames-dir", type=Path, required=True)
    prepare_adults.add_argument("--output-dir", type=Path, required=True)
    prepare_adults.add_argument("--image-size", type=int, default=320)
    prepare_adults.add_argument("--seed", type=int, default=20260730)
    prepare_adults.add_argument(
        "--max-minority-repeats",
        type=int,
        default=1,
        help=(
            "Maximum minority-class repetition. The adult default downsamples "
            "the majority instead of duplicating weak positives."
        ),
    )
    prepare_adults.add_argument("--keep-isolated-labels", action="store_true")
    prepare_adults.add_argument("--overwrite", action="store_true")
    prepare_adults.set_defaults(handler=_prepare_yolo_adults)

    train_adults = commands.add_parser(
        "train-yolo-adults",
        help="Train the compact full-scene adult-presence classifier.",
    )
    train_adults.add_argument("--dataset-dir", type=Path, required=True)
    train_adults.add_argument("--output-dir", type=Path, required=True)
    train_adults.add_argument("--base-model", default="yolo26n-cls.pt")
    train_adults.add_argument("--image-size", type=int, default=320)
    train_adults.add_argument("--batch-size", type=int, default=16)
    train_adults.add_argument("--epochs", type=int, default=30)
    train_adults.add_argument("--patience", type=int, default=6)
    train_adults.add_argument("--learning-rate", type=float, default=0.001)
    train_adults.add_argument("--device", default="mps")
    train_adults.add_argument("--seed", type=int, default=20260730)
    train_adults.add_argument("--workers", type=int, default=0)
    train_adults.add_argument("--overwrite", action="store_true")
    train_adults.set_defaults(handler=_train_yolo_adults)

    evaluate_adults = commands.add_parser(
        "evaluate-yolo-adults",
        help="Calibrate and score adult presence on untouched location/day holdouts.",
    )
    evaluate_adults.add_argument("--dataset-dir", type=Path, required=True)
    evaluate_adults.add_argument("--artifact-dir", type=Path, required=True)
    evaluate_adults.add_argument("--device", default="cpu")
    evaluate_adults.add_argument("--batch-size", type=int, default=16)
    evaluate_adults.set_defaults(handler=_evaluate_yolo_adults)

    assemble_adults = commands.add_parser(
        "assemble-yolo-adults",
        help="Package a reviewed, gate-passing adult-presence classifier.",
    )
    assemble_adults.add_argument("--base-artifact", type=Path, required=True)
    assemble_adults.add_argument("--adult-artifact", type=Path, required=True)
    assemble_adults.add_argument("--output-dir", type=Path, required=True)
    assemble_adults.add_argument("--overwrite", action="store_true")
    assemble_adults.set_defaults(handler=_assemble_yolo_adults)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
