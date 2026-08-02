from __future__ import annotations

import json
import pickle
from pathlib import Path

from app.tools.reconcile_paddleocr_resume_best import main
from scorescan.util import sha256_file


def _write_state(
    path: Path,
    *,
    epoch: int,
    accuracy: float,
    normalized_edit: float,
) -> None:
    path.write_bytes(
        pickle.dumps(
            {
                "best_model_dict": {
                    "acc": accuracy,
                    "norm_edit_dis": normalized_edit,
                    "best_epoch": epoch,
                    "is_float16": False,
                },
                "epoch": epoch,
                "global_step": 200,
            },
            protocol=2,
        )
    )


def test_reconciles_only_demonstrably_degraded_resume_best(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "latest"
    best = tmp_path / "best_accuracy"
    latest.with_suffix(".pdparams").write_bytes(b"trusted-latest")
    best.with_suffix(".pdparams").write_bytes(b"degraded-best")
    _write_state(
        latest.with_suffix(".states"),
        epoch=2,
        accuracy=0.97,
        normalized_edit=0.99,
    )
    _write_state(
        best.with_suffix(".states"),
        epoch=3,
        accuracy=0.90,
        normalized_edit=0.95,
    )
    evaluation = tmp_path / "latest-eval.log"
    evaluation.write_text(
        "acc:0.96\nnorm_edit_dis:0.985\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"

    assert main(
        [
            "--latest-prefix",
            str(latest),
            "--best-prefix",
            str(best),
            "--evaluation-log",
            str(evaluation),
            "--expected-latest-sha256",
            sha256_file(latest.with_suffix(".pdparams")),
            "--expected-epoch",
            "2",
            "--minimum-accuracy",
            "0.95",
            "--minimum-normalized-edit",
            "0.98",
            "--output-report",
            str(report),
            "--execute",
        ]
    ) == 0

    assert best.with_suffix(".pdparams").read_bytes() == b"trusted-latest"
    latest_state = pickle.loads(latest.with_suffix(".states").read_bytes())
    assert latest_state["best_model_dict"]["acc"] == 0.96
    assert latest_state["best_model_dict"]["norm_edit_dis"] == 0.985
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["executed"]
    assert payload["reconciled_best_sha256"] == payload["latest"]["sha256"]
