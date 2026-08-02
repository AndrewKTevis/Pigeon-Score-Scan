from __future__ import annotations

from app.tools.prepare_olimpic_mixed_training import (
    lmx_family_counts,
    select_family_priority_paths,
    select_group_balanced_paths,
)


def test_selection_is_deterministic_and_group_balanced() -> None:
    paths = [
        "samples/a/p1-s1",
        "samples/a/p1-s2",
        "samples/b/p1-s1",
        "samples/b/p1-s2",
        "samples/c/p1-s1",
        "samples/c/p1-s2",
    ]
    first = select_group_balanced_paths(paths, 4, "seed")
    second = select_group_balanced_paths(list(reversed(paths)), 4, "seed")
    assert first == second
    assert len({path.split("/")[1] for path in first[:3]}) == 3


def test_selection_rejects_bad_count() -> None:
    try:
        select_group_balanced_paths(["samples/a/p1-s1"], 2)
    except ValueError as error:
        assert "Cannot select" in str(error)
    else:
        raise AssertionError("Expected a ValueError")


def test_selection_rejects_bad_path() -> None:
    try:
        select_group_balanced_paths(["not-a-sample"], 1)
    except ValueError as error:
        assert "Unexpected" in str(error)
    else:
        raise AssertionError("Expected a ValueError")


def test_lmx_family_counts_cover_relations_and_marks() -> None:
    counts = lmx_family_counts(
        "C4 quarter slur:start tied:start beam:begin accent "
        "D4 eighth slur:stop tied:stop beam:end trill-mark"
    )
    assert counts == {
        "tie": 2,
        "slur": 2,
        "ornament": 1,
        "articulation": 1,
        "beam": 2,
    }


def test_family_priority_preserves_work_balance_and_prefers_rare_marks() -> None:
    paths = [
        "samples/a/plain-1",
        "samples/a/tie-1",
        "samples/b/plain-1",
        "samples/b/slur-1",
    ]
    labels = {
        "samples/a/plain-1": "C4 quarter",
        "samples/a/tie-1": "C4 quarter tied:start D4 quarter tied:stop",
        "samples/b/plain-1": "E4 quarter",
        "samples/b/slur-1": "E4 quarter slur:start F4 quarter slur:stop",
    }
    selected = select_family_priority_paths(
        paths,
        2,
        labels.__getitem__,
        "seed",
    )
    assert set(selected) == {"samples/a/tie-1", "samples/b/slur-1"}
    assert {path.split("/")[1] for path in selected} == {"a", "b"}
