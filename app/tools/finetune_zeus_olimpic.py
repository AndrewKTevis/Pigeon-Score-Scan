#!/usr/bin/env python3
"""GPU fine-tuning and leakage-safe evaluation for the OLiMPiC Zeus model.

This script intentionally keeps the published OLiMPiC candidate-test split out
of epoch-by-epoch model selection. It is not a final product benchmark; it is
only an upstream candidate test set and is opened once when explicitly asked.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def qualifies_for_selection(
    *,
    current_ser: float,
    best_ser: float,
    baseline_ser: float,
    minimum_improvement: float,
) -> bool:
    """Reject calibration changes too small to exceed observed GPU run noise."""

    values = (current_ser, best_ser, baseline_ser, minimum_improvement)
    if not all(math.isfinite(value) for value in values):
        return False
    if minimum_improvement < 0:
        raise ValueError("minimum SER improvement must be non-negative")
    return (
        current_ser < best_ser
        and current_ser <= baseline_ser - minimum_improvement
    )


_DURATION_TOKENS = frozenset(
    {
        "maxima",
        "long",
        "breve",
        "whole",
        "half",
        "quarter",
        "eighth",
        "16th",
        "32nd",
        "64th",
        "128th",
        "256th",
        "512th",
        "1024th",
    }
)
_ARTICULATION_TOKENS = frozenset(
    {
        "accent",
        "breath-mark",
        "caesura",
        "detached-legato",
        "doit",
        "falloff",
        "marcato",
        "plop",
        "scoop",
        "spiccato",
        "staccatissimo",
        "staccato",
        "stress",
        "strong-accent",
        "tenuto",
        "unstress",
    }
)
_ORNAMENT_TOKENS = frozenset(
    {
        "delayed-inverted-turn",
        "delayed-turn",
        "inverted-mordent",
        "inverted-turn",
        "mordent",
        "schleifer",
        "shake",
        "tremolo",
        "trill-mark",
        "turn",
        "wavy-line:start",
        "wavy-line:stop",
    }
)
_ACCIDENTAL_TOKENS = frozenset(
    {
        "double-flat",
        "double-sharp",
        "flat",
        "flat-flat",
        "natural",
        "natural-flat",
        "natural-sharp",
        "quarter-flat",
        "quarter-sharp",
        "sharp",
        "sharp-sharp",
        "three-quarters-flat",
        "three-quarters-sharp",
    }
)
_FAMILY_NAMES = (
    "pitch",
    "rhythm",
    "slur",
    "tie",
    "beam",
    "articulation",
    "ornament",
    "accidental",
    "attributes",
)


def _is_pitch_token(token: str) -> bool:
    return len(token) >= 2 and token[0] in "ABCDEFG" and token[1:].isdigit()


def _token_in_family(token: str, family: str) -> bool:
    if family == "pitch":
        return _is_pitch_token(token)
    if family == "rhythm":
        return (
            token in _DURATION_TOKENS
            or token in {"rest", "chord", "dot", "backup", "forward"}
            or token.startswith(("actual-notes:", "normal-notes:", "tuplet:"))
        )
    if family == "slur":
        return token.startswith("slur:")
    if family == "tie":
        return token.startswith(("tie:", "tied:"))
    if family == "beam":
        return token.startswith("beam:")
    if family == "articulation":
        return token in _ARTICULATION_TOKENS
    if family == "ornament":
        return token in _ORNAMENT_TOKENS
    if family == "accidental":
        return token in _ACCIDENTAL_TOKENS
    if family == "attributes":
        return token.startswith(("clef:", "key:", "time", "beats:", "beat-type:"))
    raise ValueError(f"unknown token family: {family}")


def _family_views(sequence: str, family: str) -> tuple[list[str], list[str]]:
    tokens = sequence.split()
    values: list[str] = []
    positioned: list[str] = []
    event_index = -1
    for token in tokens:
        if token == "rest" or _is_pitch_token(token):
            event_index += 1
        if _token_in_family(token, family):
            values.append(token)
            positioned.append(f"{event_index}:{token}")
    return values, positioned


def _sequence_distance(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_value in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_value in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def token_family_metrics(
    gold: list[str], predictions: list[str]
) -> dict[str, dict[str, float | int]]:
    if len(gold) != len(predictions):
        raise ValueError("gold/prediction sequence count differs")
    result: dict[str, dict[str, float | int]] = {}
    for family in _FAMILY_NAMES:
        reference_count = 0
        predicted_count = 0
        true_positive = 0
        positioned_true_positive = 0
        edit_distance = 0
        for expected, predicted in zip(gold, predictions, strict=True):
            expected_values, expected_positioned = _family_views(expected, family)
            predicted_values, predicted_positioned = _family_views(predicted, family)
            expected_counter = Counter(expected_values)
            predicted_counter = Counter(predicted_values)
            positioned_expected_counter = Counter(expected_positioned)
            positioned_predicted_counter = Counter(predicted_positioned)
            reference_count += len(expected_values)
            predicted_count += len(predicted_values)
            true_positive += sum(
                (expected_counter & predicted_counter).values()
            )
            positioned_true_positive += sum(
                (positioned_expected_counter & positioned_predicted_counter).values()
            )
            edit_distance += _sequence_distance(expected_values, predicted_values)

        def scores(matches: int) -> tuple[float, float, float]:
            precision = matches / predicted_count if predicted_count else (
                1.0 if not reference_count else 0.0
            )
            recall = matches / reference_count if reference_count else (
                1.0 if not predicted_count else 0.0
            )
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            return precision, recall, f1

        precision, recall, f1 = scores(true_positive)
        positioned_precision, positioned_recall, positioned_f1 = scores(
            positioned_true_positive
        )
        result[family] = {
            "reference_tokens": reference_count,
            "predicted_tokens": predicted_count,
            "precision_percent": precision * 100.0,
            "recall_percent": recall * 100.0,
            "f1_percent": f1 * 100.0,
            "positioned_precision_percent": positioned_precision * 100.0,
            "positioned_recall_percent": positioned_recall * 100.0,
            "positioned_f1_percent": positioned_f1 * 100.0,
            "filtered_token_error_rate_percent": (
                edit_distance / reference_count * 100.0
                if reference_count
                else float(edit_distance > 0) * 100.0
            ),
        }
    return result


def family_regression_gate(
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    maximum_regression: float,
) -> tuple[bool, dict[str, float]]:
    """Require every represented core notation family to remain stable."""

    if not math.isfinite(maximum_regression) or maximum_regression < 0:
        raise ValueError("maximum family regression must be non-negative")
    baseline_families = baseline_metrics.get("families", {})
    candidate_families = candidate_metrics.get("families", {})
    regressions: dict[str, float] = {}
    for family in _FAMILY_NAMES:
        baseline = baseline_families.get(family)
        candidate = candidate_families.get(family)
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            return False, {"missing_family_metrics": float("inf")}
        if int(baseline.get("reference_tokens", 0)) <= 0:
            continue
        baseline_f1 = float(baseline["positioned_f1_percent"])
        candidate_f1 = float(candidate["positioned_f1_percent"])
        if not math.isfinite(baseline_f1) or not math.isfinite(candidate_f1):
            return False, {family: float("inf")}
        regressions[family] = baseline_f1 - candidate_f1
    return (
        all(regression <= maximum_regression for regression in regressions.values()),
        regressions,
    )


def load_and_validate_manifest(prepared_dir: Path) -> dict[str, Any]:
    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    required = {"train", "calibration", "candidate_test"}
    if set(splits) != required:
        raise ValueError(
            f"Expected exactly {sorted(required)} splits, got {sorted(splits)}"
        )

    overlap = manifest.get("source_group_overlap", {})
    if any(int(value) != 0 for value in overlap.values()):
        raise ValueError(f"Source-group leakage detected: {overlap}")

    group_sets = {
        name: set(details.get("group_ids", [])) for name, details in splits.items()
    }
    for left, right in (
        ("train", "calibration"),
        ("train", "candidate_test"),
        ("calibration", "candidate_test"),
    ):
        shared = group_sets[left] & group_sets[right]
        if shared:
            raise ValueError(f"{left}/{right} share source groups: {sorted(shared)}")

    document_overlap = manifest.get("source_document_overlap")
    if document_overlap is not None:
        if not isinstance(document_overlap, dict) or any(
            int(value) != 0 for value in document_overlap.values()
        ):
            raise ValueError(
                f"Physical source-document leakage detected: {document_overlap}"
            )
        document_sets = {
            name: set(details.get("source_document_ids", []))
            for name, details in splits.items()
        }
        if any(not values for values in document_sets.values()):
            raise ValueError(
                "source-document-safe manifest is missing document ids"
            )
        for left, right in (
            ("train", "calibration"),
            ("train", "candidate_test"),
            ("calibration", "candidate_test"),
        ):
            shared_documents = document_sets[left] & document_sets[right]
            if shared_documents:
                raise ValueError(
                    f"{left}/{right} share physical source documents: "
                    f"{sorted(shared_documents)}"
                )

    for name, details in splits.items():
        pickle_path = prepared_dir / f"{name}.pickle"
        if not pickle_path.is_file():
            raise FileNotFoundError(pickle_path)
        actual_hash = sha256_file(pickle_path)
        expected_hash = details.get("pickle_sha256")
        if actual_hash != expected_hash:
            raise ValueError(
                f"{name} pickle hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return manifest


def import_upstream_zeus(upstream_dir: Path) -> ModuleType:
    zeus_file = upstream_dir / "zeus" / "zeus.py"
    if not zeus_file.is_file():
        raise FileNotFoundError(f"Missing upstream Zeus source: {zeus_file}")
    sys.path.insert(0, str(zeus_file.parent))
    sys.path.insert(0, str(upstream_dir))
    source = zeus_file.read_text(encoding="utf-8")
    # The published repository declares Python 3.10 support but imported Self
    # from typing (only available in 3.11). Keep the checkout immutable and
    # apply the minimal compatibility substitution only in memory.
    source = source.replace(
        "from typing import Self", "from typing_extensions import Self"
    )
    source = source.replace(
        "get_initial_state(batch_size=batch_size, dtype=tf.float32)",
        "get_initial_state(batch_size=batch_size, dtype=tf.float32)\n"
        "        states = tf.nest.map_structure(\n"
        "            lambda state: tf.cast("
        "state, self._target_embedding(inputs).dtype"
        "), states)",
    )
    source = source.replace(
        "outputs, new_state = cell(inputs, state)",
        "state = tf.nest.map_structure(\n"
        "                    lambda value: tf.cast(value, inputs.dtype), state\n"
        "                )\n"
        "                outputs, new_state = cell(inputs, state)",
    )
    module = ModuleType("scorescan_upstream_zeus")
    module.__file__ = str(zeus_file)
    module.__package__ = ""
    exec(compile(source, str(zeus_file), "exec"), module.__dict__)
    return module


def _stable_subset(dataset: Any, limit: int | None, seed: int) -> None:
    if limit is None or limit >= len(dataset.data):
        return
    import numpy as np

    indices = np.random.RandomState(seed).choice(
        len(dataset.data), size=limit, replace=False
    )
    dataset.data = [dataset.data[int(index)] for index in sorted(indices)]


def _prediction_strings(model: Any, dataset: Any, tags: Any) -> list[str]:
    predicted_tags = model.predict(dataset.tf_dataset(), verbose=0)
    predictions: list[str] = []
    for sequence in predicted_tags:
        predictions.append(
            " ".join(tags.tags[int(tag)] for tag in sequence.numpy())
        )
    return predictions


def _evaluate(
    *,
    model: Any,
    dataset: Any,
    tags: Any,
    ser_metric_module: ModuleType,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    predictions = _prediction_strings(model, dataset, tags)
    gold = [entry["lmx"] for entry in dataset.data]
    metrics = {
        name: float(value)
        for name, value in ser_metric_module.ser_metric(gold, predictions).items()
    }
    metrics["families"] = token_family_metrics(gold, predictions)
    (output_dir / f"{dataset.basename}.{label}.lmx").write_text(
        "\n".join(predictions) + "\n", encoding="utf-8"
    )
    (output_dir / f"{dataset.basename}.{label}.metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def _build_args(
    *,
    base_model: Path,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    augment: str,
    seed: int,
) -> SimpleNamespace:
    options = json.loads((base_model / "options.json").read_text(encoding="utf-8"))
    options.update(
        {
            "augment": augment,
            "batch_size": batch_size,
            "decay": "none",
            "epochs": epochs,
            "evaluation_each": 1,
            "evaluation_from": 1,
            "exp": output_dir.name,
            "learning_rate": learning_rate,
            "load": None,
            "logdir": str(output_dir),
            "max_predict_length": max(900, int(options["max_predict_length"])),
            "max_train_length": max(900, int(options["max_train_length"])),
            "max_width": options.get("max_width"),
            "seed": seed,
            "threads": 0,
            "train": "prepared/olimpic-real-v1/train",
            "verbose": 2,
            "visualize_only": False,
        }
    )
    return SimpleNamespace(**options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--minimum-ser-improvement",
        type=float,
        default=0.10,
        help="minimum absolute calibration SER percentage-point gain for selection",
    )
    parser.add_argument(
        "--maximum-family-f1-regression",
        type=float,
        default=0.50,
        help="maximum positioned F1 percentage-point regression in any core family",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "mixed_float16"),
        default="float32",
    )
    parser.add_argument(
        "--augment",
        default="h:8,rotate:0.5,v:4,de,en3:0.12,n:0.006,c:-0.5:0.5,b:-0.2:0.15",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-calibration-samples", type=int)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--evaluate-candidate-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    started = time.time()
    for directory in (cli.upstream_dir, cli.base_model, cli.prepared_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    for name in ("options.json", "tags.txt", "weights.h5"):
        if not (cli.base_model / name).is_file():
            raise FileNotFoundError(cli.base_model / name)

    manifest = load_and_validate_manifest(cli.prepared_dir)
    cli.output_dir.mkdir(parents=True, exist_ok=False)

    # TensorFlow 2.15 uses its bundled Keras 2 implementation. Newer runtimes
    # may opt into tf-keras explicitly by setting TF_USE_LEGACY_KERAS before
    # launching this process.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    zeus = import_upstream_zeus(cli.upstream_dir)
    tf = zeus.tf

    tf.keras.utils.set_random_seed(cli.seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        with contextlib.suppress(RuntimeError):
            tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus and not cli.allow_cpu:
        raise RuntimeError("No TensorFlow GPU detected; refusing silent CPU training")
    if cli.precision == "mixed_float16":
        if not gpus:
            raise RuntimeError("mixed_float16 requires a GPU in this training path")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    args = _build_args(
        base_model=cli.base_model,
        output_dir=cli.output_dir,
        batch_size=cli.batch_size,
        epochs=cli.epochs,
        learning_rate=cli.learning_rate,
        augment=cli.augment,
        seed=cli.seed,
    )
    base_tags = zeus.LMXDataset.from_tags(str(cli.base_model / "tags.txt"))
    train = zeus.LMXDataset(
        str(cli.prepared_dir / "train"), args, train_dataset=base_tags
    )
    calibration = zeus.LMXDataset(
        str(cli.prepared_dir / "calibration"), args, train_dataset=base_tags
    )
    candidate_test = None
    if cli.evaluate_candidate_test:
        candidate_test = zeus.LMXDataset(
            str(cli.prepared_dir / "candidate_test"),
            args,
            train_dataset=base_tags,
        )
    _stable_subset(train, cli.max_train_samples, cli.seed)
    _stable_subset(calibration, cli.max_calibration_samples, cli.seed + 1)

    args.train_batches = math.ceil(len(train.data) / args.batch_size)
    model = zeus.Model(args, base_tags)
    dummy = tf.RaggedTensor.from_tensor(
        tf.ones([1, args.height, 128, 1], dtype=tf.float32), ragged_rank=2
    )
    model.decoder_inference(model.encoder(dummy), tf.constant(1))
    model.built = True
    model.load_weights(str(cli.base_model / "weights.h5"))
    model.optimizer.learning_rate.assign(cli.learning_rate)

    shutil.copy2(cli.base_model / "tags.txt", cli.output_dir / "tags.txt")
    run_options = dict(vars(args))
    run_options.update(
        {
            "base_model": str(cli.base_model),
            "prepared_dir": str(cli.prepared_dir),
            "upstream_dir": str(cli.upstream_dir),
        }
    )
    (cli.output_dir / "options.json").write_text(
        json.dumps(run_options, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics: dict[str, Any] = {
        "calibration_baseline": _evaluate(
            model=model,
            dataset=calibration,
            tags=base_tags,
            ser_metric_module=zeus.ser_metric,
            output_dir=cli.output_dir,
            label="baseline",
        ),
        "epochs": [],
    }

    baseline_ser = float(metrics["calibration_baseline"]["SER"])
    best_ser = baseline_ser
    best_epoch = 0
    best_weights = cli.output_dir / "weights.best.h5"
    model.save_weights(str(best_weights))

    if not cli.baseline_only:

        class CalibrationEvaluator(tf.keras.callbacks.Callback):
            def on_epoch_end(
                self, epoch: int, logs: dict[str, Any] | None = None
            ) -> None:
                nonlocal best_epoch, best_ser
                epoch_number = epoch + 1
                epoch_metrics = _evaluate(
                    model=model,
                    dataset=calibration,
                    tags=base_tags,
                    ser_metric_module=zeus.ser_metric,
                    output_dir=cli.output_dir,
                    label=f"epoch-{epoch_number:03d}",
                )
                record = {
                    "epoch": epoch_number,
                    "fit": {
                        name: float(value)
                        for name, value in (logs or {}).items()
                    },
                    "calibration": epoch_metrics,
                }
                metrics["epochs"].append(record)
                current_ser = float(epoch_metrics["SER"])
                ser_eligible = qualifies_for_selection(
                    current_ser=current_ser,
                    best_ser=best_ser,
                    baseline_ser=baseline_ser,
                    minimum_improvement=cli.minimum_ser_improvement,
                )
                family_eligible, family_regressions = family_regression_gate(
                    baseline_metrics=metrics["calibration_baseline"],
                    candidate_metrics=epoch_metrics,
                    maximum_regression=cli.maximum_family_f1_regression,
                )
                record["selection"] = {
                    "ser_eligible": ser_eligible,
                    "family_eligible": family_eligible,
                    "family_positioned_f1_regressions": family_regressions,
                }
                if ser_eligible and family_eligible:
                    best_ser = current_ser
                    best_epoch = epoch_number
                    model.save_weights(str(best_weights))
                (cli.output_dir / "metrics.partial.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

        model.fit(
            train.tf_dataset(training=True),
            epochs=cli.epochs,
            callbacks=[CalibrationEvaluator()],
            verbose=args.verbose,
        )

    model.load_weights(str(best_weights))
    metrics["best_epoch"] = best_epoch
    metrics["best_calibration_ser_percent"] = best_ser
    metrics["calibration_best"] = _evaluate(
        model=model,
        dataset=calibration,
        tags=base_tags,
        ser_metric_module=zeus.ser_metric,
        output_dir=cli.output_dir,
        label="best",
    )
    if candidate_test is not None:
        metrics["candidate_test_best"] = _evaluate(
            model=model,
            dataset=candidate_test,
            tags=base_tags,
            ser_metric_module=zeus.ser_metric,
            output_dir=cli.output_dir,
            label="best",
        )

    shutil.copy2(best_weights, cli.output_dir / "weights.h5")
    report = {
        "format": 1,
        "purpose": "OLiMPiC real-scan Zeus fine-tuning; not final product acceptance",
        "runtime": {
            "cuda_built": bool(tf.test.is_built_with_cuda()),
            "gpu_devices": [gpu.name for gpu in gpus],
            "keras_policy": tf.keras.mixed_precision.global_policy().name,
            "host": platform.platform(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
        },
        "data": {
            "manifest_sha256": sha256_file(cli.prepared_dir / "manifest.json"),
            "manifest_name": manifest["name"],
            "source_document_isolation_verified": (
                isinstance(manifest.get("source_document_overlap"), dict)
                and not any(
                    int(value) != 0
                    for value in manifest["source_document_overlap"].values()
                )
            ),
            "source_document_overlap": manifest.get(
                "source_document_overlap"
            ),
            "train_samples_used": len(train.data),
            "calibration_samples_used": len(calibration.data),
            "candidate_test_opened": candidate_test is not None,
            "candidate_test_is_final_product_benchmark": False,
        },
        "artifacts": {
            "base_weights_sha256": sha256_file(cli.base_model / "weights.h5"),
            "fine_tuned_weights_sha256": sha256_file(cli.output_dir / "weights.h5"),
            "tags_sha256": sha256_file(cli.output_dir / "tags.txt"),
        },
        "configuration": {
            "augment": cli.augment,
            "batch_size": cli.batch_size,
            "epochs": 0 if cli.baseline_only else cli.epochs,
            "learning_rate": cli.learning_rate,
            "minimum_ser_improvement": cli.minimum_ser_improvement,
            "maximum_family_f1_regression": cli.maximum_family_f1_regression,
            "precision": cli.precision,
            "seed": cli.seed,
        },
        "metrics_percent": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (cli.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with contextlib.suppress(FileNotFoundError):
        (cli.output_dir / "metrics.partial.json").unlink()
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
