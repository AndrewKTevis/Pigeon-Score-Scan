from __future__ import annotations

import math
import random
import sys
import json
from pathlib import Path

import numpy as np
import paddle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEOCR_ROOT = (
    PROJECT_ROOT
    / "training_data"
    / "external"
    / "corpora"
    / "paddleocr_2661c7c0"
)
sys.path.insert(0, str(PADDLEOCR_ROOT))

from ppocr.data.imaug.make_border_map import MakeBorderMap  # noqa: E402
from ppocr.data.imaug.make_shrink_map import MakeShrinkMap  # noqa: E402
from ppocr.data.imaug.random_crop_data import RandomCrop  # noqa: E402
from ppocr.data.imaug.label_ops import DetLabelEncode  # noqa: E402
from ppocr.losses.det_db_loss import DBLoss  # noqa: E402


def main() -> int:
    paddle.set_device("cpu")
    random.seed(17)
    sample = {
        "image": np.full((2400, 2400, 3), 255, dtype=np.uint8),
        "label": json.dumps(
            [
                {
                    "transcription": "Allegro",
                    "points": [
                        [50, 50],
                        [250, 50],
                        [250, 150],
                        [50, 150],
                    ],
                    "hard_negative_sampling_authorized": True,
                }
            ]
        ),
    }
    sample = DetLabelEncode()(sample)
    if sample is None:
        raise RuntimeError("hard-negative label encoding returned no sample")
    if sample.get("hard_negative_sampling_authorized") is not True:
        raise RuntimeError("hard-negative authorization was not propagated")
    sample = RandomCrop(
        size=(640, 640),
        max_tries=80,
        keep_ratio=True,
        negative_crop_prob=1.0,
    )(sample)
    sample = MakeBorderMap()(sample)
    sample = MakeShrinkMap()(sample)
    if len(sample["polys"]) != 0:
        raise RuntimeError("hard-negative crop unexpectedly retained text")

    shape = (1, 640, 640)
    predictions = {
        "maps": paddle.full((1, 3, 640, 640), 0.25, dtype="float32")
    }
    labels = [
        paddle.zeros((1, 3, 640, 640), dtype="float32"),
        paddle.to_tensor(sample["threshold_map"])[None, :, :],
        paddle.to_tensor(sample["threshold_mask"])[None, :, :],
        paddle.to_tensor(sample["shrink_map"])[None, :, :],
        paddle.to_tensor(sample["shrink_mask"])[None, :, :],
    ]
    losses = DBLoss(
        main_loss_type="DiceFocalLoss",
        focal_alpha=0.25,
        focal_gamma=2.5,
    )(predictions, labels)
    loss_values = np.asarray(losses["loss"].numpy()).reshape(-1)
    if loss_values.size != 1:
        raise RuntimeError(f"unexpected hard-negative loss shape: {loss_values.shape}")
    loss = float(loss_values[0])
    if not math.isfinite(loss) or loss <= 0:
        raise RuntimeError(f"invalid hard-negative loss: {loss}")
    if tuple(sample["shrink_map"].shape) != shape[1:]:
        raise RuntimeError("unexpected hard-negative target shape")
    print(f"hard-negative crop/loss smoke test passed: loss={loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
