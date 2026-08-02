from __future__ import annotations

from pathlib import Path

from app.tools.humdrum_boundary import analyze_humdrum_boundary


def _write(tmp_path: Path, body: str, *, ain: str = "") -> Path:
    path = tmp_path / "fixture.krn"
    path.write_text(
        f"!!!AIN: {ain}\n{body}",
        encoding="utf-8",
    )
    return path


def test_accepts_complex_piano_with_voice_split(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern\t**dynam
*part1\t*part1\t*part1
*staff2\t*staff1\t*
*Ipiano\t*Ipiano\t*
*M4/4\t*M4/4\t*
=1\t=1\t=1
*^\t*\t*
4C\t4E\t4c\t.
*v\t*v\t*\t*
4D\t4d\t.
*-\t*-\t*-
""",
        ain="1 piano",
    )
    result = analyze_humdrum_boundary(path, instrumentation="1 piano")
    assert result.accepted is True
    assert result.score_shape == "keyboard"
    assert result.physical_staff_count == 2
    assert result.maximum_voices_per_keyboard_staff == 2


def test_accepts_keyboard_plus_independent_ensemble_timelines(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern\t**kern\t**kern
*part3\t*part3\t*part2\t*part1
*staff4\t*staff3\t*staff2\t*staff1
*Ipiano\t*Ipiano\t*Icello\t*Ivioln
*M3/4\t*M3/4\t*M3/4\t*M3/4
=1\t=1\t=1\t=1
4C\t4c\t2G\t8g
4D\t4d\t.\t8a
4E\t4e\t4A\t4b
*-\t*-\t*-\t*-
""",
        ain="1 piano 1 cello 1 violn",
    )
    result = analyze_humdrum_boundary(
        path,
        instrumentation="1 piano 1 cello 1 violn",
    )
    assert result.accepted is True
    assert result.score_shape == "keyboard_plus_single_staff_ensemble"
    assert result.part_staff_counts == (1, 1, 2)


def test_rejects_non_keyboard_voice_split_and_lyrics(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**text
*part1\t*part1
*staff1\t*staff1
*Iviolin\t*
*M4/4\t*
=1\t=1
*^\t*
4c\t4e\tla
*v\t*v\t*
*-\t*-
""",
        ain="1 violin",
    )
    result = analyze_humdrum_boundary(path, instrumentation="1 violin")
    assert result.accepted is False
    assert "lyrics_harmony_or_figured_bass" in result.reasons
    assert (
        "more_than_one_independent_voice_per_non_keyboard_staff"
        in result.reasons
    )


def test_rejects_true_polymeter_and_percussion(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern
*part2\t*part1
*staff2\t*staff1
*Itimpa\t*Ivioln
*M3/4\t*M4/4
=1\t=1
4C\t4c
*-\t*-
""",
        ain="1 timpa 1 violn",
    )
    result = analyze_humdrum_boundary(
        path,
        instrumentation="1 timpa 1 violn",
    )
    assert "true_polymeter" in result.reasons
    assert "unpitched_or_percussion_notation" in result.reasons


def test_rejects_missing_topology_instead_of_guessing(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern
*M4/4
=1
4c
*-
""",
        ain="1 violin",
    )
    result = analyze_humdrum_boundary(path, instrumentation="1 violin")
    assert result.accepted is False
    assert "missing_part_or_staff_topology" in result.reasons


def test_infers_only_one_keyboard_part_when_every_spine_has_staff(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern\t**dynam
*staff2\t*staff1\t*
*Ipiano\t*Ipiano\t*
*M4/4\t*M4/4\t*
=1\t=1\t=1
4C\t4c\tp
*-\t*-\t*-
""",
        ain="1 piano",
    )
    result = analyze_humdrum_boundary(path, instrumentation="1 piano")

    assert result.accepted is True
    assert result.topology_complete is True
    assert result.score_shape == "keyboard"
    assert result.part_count == 1
    assert result.keyboard_part_count == 1
    assert result.part_staff_counts == (2,)


def test_implicit_part_inference_still_rejects_missing_keyboard_staff(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern
*Ipiano
4c
*-
""",
        ain="1 piano",
    )
    result = analyze_humdrum_boundary(path, instrumentation="1 piano")

    assert result.accepted is False
    assert "missing_part_or_staff_topology" in result.reasons


def test_spine_exchange_preserves_staff_voice_accounting(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern
*part2\t*part1
*staff2\t*staff1
*Icello\t*Ivioln
*x\t*x
4c\t4C
*-\t*-
""",
        ain="1 cello 1 violin",
    )
    result = analyze_humdrum_boundary(
        path,
        instrumentation="1 cello 1 violin",
    )
    assert result.accepted is True
    assert result.score_shape == "single_staff_ensemble"


def test_accepts_preloaded_source_lines_without_rereading(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "not-on-disk.krn"
    result = analyze_humdrum_boundary(
        missing_path,
        instrumentation="1 violin",
        source_lines=(
            "**kern",
            "*part1",
            "*staff1",
            "*Iviolin",
            "4c",
            "*-",
        ),
    )
    assert result.accepted is True


def test_normalizes_two_encoded_parts_for_one_keyboard_instrument(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern
*part2\t*part1
*staff2\t*staff1
*Iorgan\t*Iempty
4C\t4c
*-\t*-
""",
        ain="1 empty 1 organ",
    )
    result = analyze_humdrum_boundary(
        path,
        instrumentation="1 empty 1 organ",
    )
    assert result.accepted is True
    assert result.score_shape == "keyboard"
    assert result.part_count == 1
    assert result.keyboard_part_count == 1
    assert result.part_staff_counts == (2,)


def test_rejects_unidentified_part_in_mixed_instrument_score(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern\t**kern
*part3\t*part2\t*part1
*staff3\t*staff2\t*staff1
*Ipiano\t*Ipiano\t*Iempty
4C\t4c\t4g
*-\t*-\t*-
""",
        ain="1 piano 1 violin",
    )
    result = analyze_humdrum_boundary(
        path,
        instrumentation="1 piano 1 violin",
    )
    assert result.accepted is False
    assert "missing_part_instrument_identity" in result.reasons


def test_does_not_collapse_two_keyboard_instruments(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
**kern\t**kern
*part2\t*part1
*staff2\t*staff1
*Ipiano\t*Ipiano
4C\t4c
*-\t*-
""",
        ain="2 piano",
    )
    result = analyze_humdrum_boundary(
        path,
        instrumentation="2 piano",
    )
    assert result.accepted is False
    assert "more_than_one_keyboard_part" in result.reasons
