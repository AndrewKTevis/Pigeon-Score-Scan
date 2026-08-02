from __future__ import annotations

"""Inference helpers for independently evaluating the published PrIMuS model.

This module does not make the third-party model part of ScoreScan.  It only keeps
the preprocessing and CTC decoding used by the reproducible benchmark in one
tested place.
"""

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_primus_image(path: Path, height: int = 128) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"unable to read image: {path}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"invalid image dimensions: {path}")
    width = max(1, int(float(height * image.shape[1]) / image.shape[0]))
    resized = cv2.resize(image, (width, height))
    normalized = (255.0 - resized.astype(np.float32)) / 255.0
    return normalized.reshape(1, height, width, 1)


def greedy_ctc_decode(logits: np.ndarray, vocabulary: Sequence[str]) -> tuple[str, ...]:
    """Decode time-major CTC logits using TensorFlow's greedy-decoder rules."""

    if logits.ndim != 3 or logits.shape[1] != 1:
        raise ValueError(f"expected [time, 1, classes] logits, got {logits.shape}")
    blank = len(vocabulary)
    if logits.shape[2] != blank + 1:
        raise ValueError(
            f"logit class count {logits.shape[2]} does not match "
            f"vocabulary size {len(vocabulary)} plus blank"
        )

    decoded: list[str] = []
    previous = -1
    for value in np.argmax(logits[:, 0, :], axis=1):
        index = int(value)
        if index != blank and index != previous:
            decoded.append(vocabulary[index])
        previous = index
    return tuple(decoded)


class PrimusOnnxModel:
    def __init__(self, model_path: Path, vocabulary_path: Path) -> None:
        self.model_path = model_path
        self.vocabulary_path = vocabulary_path
        self.vocabulary = tuple(
            line.strip()
            for line in vocabulary_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not self.vocabulary:
            raise ValueError(f"empty vocabulary: {vocabulary_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_names = {item.name for item in self.session.get_inputs()}
        if input_names != {"model_input:0", "keep_prob:0"}:
            raise ValueError(f"unexpected model inputs: {sorted(input_names)}")
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ValueError(f"expected one model output, got {len(outputs)}")
        self.output_name = outputs[0].name

    def predict_tokens(self, image_path: Path) -> tuple[str, ...]:
        image = preprocess_primus_image(image_path)
        (logits,) = self.session.run(
            [self.output_name],
            {
                "model_input:0": image,
                "keep_prob:0": np.asarray(1.0, dtype=np.float32),
            },
        )
        return greedy_ctc_decode(logits, self.vocabulary)
