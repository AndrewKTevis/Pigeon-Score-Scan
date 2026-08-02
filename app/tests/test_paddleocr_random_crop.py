from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PADDLE_PATCH_ROOT = PROJECT_ROOT / "third_party" / "paddleocr-patches"
RANDOM_CROP_SOURCE = (
    PADDLE_PATCH_ROOT
    / "ppocr"
    / "data"
    / "imaug"
    / "random_crop_data.py"
)
MAKE_BORDER_MAP_SOURCE = RANDOM_CROP_SOURCE.with_name("make_border_map.py")
MAKE_SHRINK_MAP_SOURCE = RANDOM_CROP_SOURCE.with_name("make_shrink_map.py")
DETECTION_CONFIG = PROJECT_ROOT / "training" / "ppocrv6_scorescan_det.yml"


def _load_class(
    source: Path,
    class_name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.get_device = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    spec = importlib.util.spec_from_file_location(
        f"scorescan_test_{source.stem}",
        source,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _load_random_crop_class(monkeypatch: pytest.MonkeyPatch):
    return _load_class(RANDOM_CROP_SOURCE, "RandomCrop", monkeypatch)


def _sample(
    *,
    hard_negative_sampling_authorized: bool = False,
) -> dict[str, object]:
    sample: dict[str, object] = {
        "image": np.full((2400, 2400, 3), 255, dtype=np.uint8),
        "polys": np.array(
            [[[50, 50], [250, 50], [250, 150], [50, 150]]],
            dtype=np.float32,
        ),
        "ignore_tags": [False],
        "texts": ["Allegro"],
    }
    if hard_negative_sampling_authorized:
        sample["hard_negative_sampling_authorized"] = True
    return sample


def test_random_crop_can_produce_notation_only_hard_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_crop = _load_random_crop_class(monkeypatch)
    random.seed(17)
    transform = random_crop(
        size=(640, 640),
        max_tries=80,
        keep_ratio=True,
        negative_crop_prob=1.0,
    )

    output = transform(
        _sample(hard_negative_sampling_authorized=True)
    )

    assert output["image"].shape == (640, 640, 3)
    assert output["polys"].shape[0] == 0
    assert output["ignore_tags"] == []
    assert output["texts"] == []

    make_border_map = _load_class(
        MAKE_BORDER_MAP_SOURCE,
        "MakeBorderMap",
        monkeypatch,
    )
    make_shrink_map = _load_class(
        MAKE_SHRINK_MAP_SOURCE,
        "MakeShrinkMap",
        monkeypatch,
    )
    output = make_border_map()(output)
    output = make_shrink_map()(output)
    assert np.count_nonzero(output["threshold_mask"]) == 0
    assert np.count_nonzero(output["shrink_map"]) == 0
    assert np.all(output["shrink_mask"] == 1)


def test_random_crop_retains_positive_training_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_crop = _load_random_crop_class(monkeypatch)
    random.seed(17)
    transform = random_crop(
        size=(640, 640),
        max_tries=80,
        keep_ratio=True,
        negative_crop_prob=0.0,
    )

    output = transform(_sample())

    assert output["image"].shape == (640, 640, 3)
    assert output["polys"].shape[0] == 1
    assert output["ignore_tags"] == [False]
    assert output["texts"] == ["Allegro"]


def test_random_crop_cannot_use_sparse_page_as_hard_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random_crop = _load_random_crop_class(monkeypatch)
    random.seed(17)
    transform = random_crop(
        size=(640, 640),
        max_tries=80,
        keep_ratio=True,
        negative_crop_prob=1.0,
    )

    output = transform(_sample())

    assert output["polys"].shape[0] == 1
    assert output["texts"] == ["Allegro"]


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_random_crop_rejects_invalid_negative_probability(
    monkeypatch: pytest.MonkeyPatch,
    probability: float,
) -> None:
    random_crop = _load_random_crop_class(monkeypatch)
    with pytest.raises(ValueError, match="negative_crop_prob"):
        random_crop(negative_crop_prob=probability)


def test_detection_training_uses_only_page_authorized_negative_regions() -> None:
    config = yaml.safe_load(DETECTION_CONFIG.read_text(encoding="utf-8"))
    transforms = config["Train"]["dataset"]["transforms"]
    random_crop = next(
        item["RandomCrop"] for item in transforms if "RandomCrop" in item
    )

    assert random_crop["size"] == [640, 640]
    assert random_crop["max_tries"] >= 40
    assert 0.0 < random_crop["negative_crop_prob"] <= 0.30
