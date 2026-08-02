from __future__ import annotations

"""Pinned model inventory for the network-free Windows runtime."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BundledModel:
    package: str
    relative_path: str
    size: int
    sha256: str


BUNDLED_MODELS = (
    BundledModel(
        "homr",
        "segmentation/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
        57_311_361,
        "6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d",
    ),
    BundledModel(
        "homr",
        "transformer/decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
        47_299_551,
        "3e10fd5ae52d0b86792721922fcd954c283a7ed365de7446425bdabe38f3e57d",
    ),
    BundledModel(
        "homr",
        "transformer/encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
        52_861_122,
        "4c16df852b3789f2676b0d49f0545dab0740e4005f7b472c5252add642f5d5eb",
    ),
    BundledModel(
        "rapidocr",
        "models/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        585_532,
        "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    ),
    BundledModel(
        "rapidocr",
        "models/PP-OCRv6_det_small.onnx",
        9_929_594,
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    ),
    BundledModel(
        "rapidocr",
        "models/PP-OCRv6_rec_small.onnx",
        21_234_383,
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundled_models(
    site_packages: Path,
    *,
    verify_hashes: bool,
) -> list[str]:
    problems: list[str] = []
    for model in BUNDLED_MODELS:
        path = site_packages / model.package / Path(model.relative_path)
        if not path.is_file():
            problems.append(f"missing: {model.package}/{model.relative_path}")
            continue
        if path.stat().st_size != model.size:
            problems.append(f"size mismatch: {model.package}/{model.relative_path}")
            continue
        if verify_hashes and sha256_file(path) != model.sha256:
            problems.append(f"SHA-256 mismatch: {model.package}/{model.relative_path}")
    return problems
