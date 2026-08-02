#!/usr/bin/env python3
"""Validate detector contracts that require the isolated PyTorch runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.train_deepscores_symbol_detector import (
    detector_microbatch_plan,
    legacy_detector_sampled_indices,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict[str, object]:
    import torch
    from torch.utils.data import WeightedRandomSampler

    torch.set_num_threads(1)
    inputs = torch.tensor([1.0, 2.0, 4.0, 8.0])
    targets = torch.tensor([0.5, -1.0, 3.0, 2.0])
    direct_weight = torch.tensor(0.75, requires_grad=True)
    ((direct_weight * inputs - targets) ** 2).mean().backward()

    micro_weight = torch.tensor(0.75, requires_grad=True)
    for start, end, fraction in detector_microbatch_plan(4, 1):
        micro_loss = (
            (micro_weight * inputs[start:end] - targets[start:end]) ** 2
        ).mean()
        (micro_loss * fraction).backward()
    gradient_delta = abs(
        float(micro_weight.grad.item()) - float(direct_weight.grad.item())
    )

    weights = [0.5, 1.0, 2.0, 4.0]
    generator = torch.Generator().manual_seed(20260729)
    epoch_state = generator.get_state()
    sampler = WeightedRandomSampler(
        weights,
        num_samples=20,
        replacement=True,
        generator=generator,
    )
    sampler_iterator = iter(sampler)
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    expected_indices = list(sampler_iterator)
    recovered_indices = legacy_detector_sampled_indices(
        weights,
        num_samples=20,
        epoch_loader_generator_state=epoch_state,
    )
    sampler_exact = recovered_indices == expected_indices
    passed = gradient_delta <= 1e-6 and sampler_exact
    source = PROJECT_ROOT / "app/tools/train_deepscores_symbol_detector.py"
    return {
        "format": 1,
        "name": "scorescan-detector-isolated-torch-contracts-v1",
        "passed": passed,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "checks": {
            "microbatch_gradient": {
                "passed": gradient_delta <= 1e-6,
                "maximum_absolute_delta": gradient_delta,
                "tolerance": 1e-6,
            },
            "legacy_sampler_recovery": {
                "passed": sampler_exact,
                "sample_count": len(expected_indices),
                "expected": expected_indices,
                "observed": recovered_indices,
            },
        },
        "input": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
