from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import pytest
import yaml


TOOL = Path(__file__).resolve().parents[1] / "tools" / "prepare_olimpic_real_training.py"
SPEC = importlib.util.spec_from_file_location("prepare_olimpic_real_training", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sample(group: int, index: int) -> str:
    return f"samples/{group}/p1-s{index}"


def test_split_is_group_isolated_and_deterministic() -> None:
    development = [
        _sample(group, index)
        for group in range(10)
        for index in range(1, 4)
    ]
    published_test = [
        _sample(group, 1)
        for group in range(20, 25)
    ]
    first = MODULE.split_development_groups(
        development,
        published_test,
        training_group_count=8,
        seed="fixed",
    )
    second = MODULE.split_development_groups(
        list(reversed(development)),
        list(reversed(published_test)),
        training_group_count=8,
        seed="fixed",
    )
    assert first == second
    train_groups = {MODULE.source_group(sample) for sample in first["train"]}
    calibration_groups = {
        MODULE.source_group(sample) for sample in first["calibration"]
    }
    test_groups = {
        MODULE.source_group(sample) for sample in first["candidate_test"]
    }
    assert len(train_groups) == 8
    assert len(calibration_groups) == 2
    assert not train_groups & calibration_groups
    assert not train_groups & test_groups
    assert not calibration_groups & test_groups


def test_published_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MODULE.split_development_groups(
            [_sample(1, 1), _sample(2, 1)],
            [_sample(2, 2)],
            training_group_count=1,
            seed="fixed",
        )


def test_physical_document_regroup_moves_contaminated_calibration_to_train() -> None:
    development = [_sample(group, 1) for group in range(6)]
    published_test = [_sample(group, 1) for group in range(20, 22)]
    score_only = MODULE.split_development_groups(
        development,
        published_test,
        training_group_count=4,
        seed="fixed",
    )
    initial_train = {
        MODULE.source_group(sample) for sample in score_only["train"]
    }
    initial_calibration = {
        MODULE.source_group(sample)
        for sample in score_only["calibration"]
    }
    contaminated = next(iter(initial_calibration))
    trained = next(iter(initial_train))
    documents = {
        str(group): f"doc-{group}"
        for group in [*range(6), 20, 21]
    }
    documents[contaminated] = documents[trained]

    isolated = MODULE.split_development_groups(
        development,
        published_test,
        training_group_count=4,
        seed="fixed",
        source_documents=documents,
    )
    train_groups = {
        MODULE.source_group(sample) for sample in isolated["train"]
    }
    calibration_groups = {
        MODULE.source_group(sample) for sample in isolated["calibration"]
    }
    assert contaminated in train_groups
    assert len(train_groups) == 5
    assert len(calibration_groups) == 1
    assert not (
        {documents[group] for group in train_groups}
        & {documents[group] for group in calibration_groups}
    )


def test_read_source_documents_requires_one_document_per_score(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    (mapping / "1.yaml").write_text(
        yaml.safe_dump(
            {
                "1/p1-s1": {"imslpDocument": "#10"},
                "1/p1-s2": {"imslpDocument": "#11"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one IMSLP document"):
        MODULE.read_source_documents(mapping)


def test_pack_split_preserves_upstream_zeus_schema(tmp_path: Path) -> None:
    sample = _sample(12, 3)
    base = tmp_path.joinpath(*Path(sample).parts)
    base.parent.mkdir(parents=True)
    base.with_suffix(".png").write_bytes(b"\x89PNG\r\n\x1a\n")
    base.with_suffix(".lmx").write_text("measure C4 quarter", encoding="utf-8")
    base.with_suffix(".musicxml").write_text(
        "<?xml version='1.0'?><score-partwise version='4.0'/>",
        encoding="utf-8",
    )
    output = tmp_path / "packed.pickle"
    report = MODULE.pack_split(tmp_path, [sample], output)
    with output.open("rb") as handle:
        entries = pickle.load(handle)
    assert entries == [
        {
            "path": sample,
            "image": b"\x89PNG\r\n\x1a\n",
            "lmx": "measure C4 quarter",
            "musicxml": "<?xml version='1.0'?><score-partwise version='4.0'/>",
        }
    ]
    assert report["samples"] == 1
    assert report["source_groups"] == 1
    assert len(report["fingerprint"]) == 64
    assert len(report["pickle_sha256"]) == 64
