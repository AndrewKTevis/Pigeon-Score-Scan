import numpy as np
import pytest

from scorescan.primus_onnx import greedy_ctc_decode


def test_greedy_ctc_decode_collapses_repeats_but_respects_blanks() -> None:
    # Vocabulary has two tokens, making class 2 the CTC blank.
    classes = [0, 0, 2, 0, 1, 1, 2]
    logits = np.full((len(classes), 1, 3), -10.0, dtype=np.float32)
    for row, value in enumerate(classes):
        logits[row, 0, value] = 10.0

    assert greedy_ctc_decode(logits, ("note", "barline")) == (
        "note",
        "note",
        "barline",
    )


def test_greedy_ctc_decode_rejects_wrong_class_count() -> None:
    with pytest.raises(ValueError, match="class count"):
        greedy_ctc_decode(np.zeros((2, 1, 4), dtype=np.float32), ("note",))
