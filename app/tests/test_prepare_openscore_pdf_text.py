from __future__ import annotations

from collections import Counter
import zipfile

import pytest

from app.tools.prepare_openscore_pdf_text import (
    ExcludedTextBoxError,
    _render_pdf,
    consume_source_text_role,
    extract_page_words,
    is_usable_text,
    map_pdf_box_to_image,
    select_source_shard,
    source_text_token_counts,
    text_token_keys,
)


def test_filters_music_fonts_and_private_use_glyphs() -> None:
    assert is_usable_text("Allegretto", "ABC+Edwin-Roman")
    assert is_usable_text("13", "ABC+Edwin-Italic")
    assert not is_usable_text("\ue0a4", "ABC+Edwin-Roman")
    assert not is_usable_text("p", "ABC+Leland")
    assert not is_usable_text(" -- ", "ABC+Edwin-Roman")
    assert is_usable_text(
        " = ",
        "ABC+Edwin-Roman",
        include_punctuation=True,
    )
    assert not is_usable_text(
        "\ue0a4",
        "ABC+Edwin-Roman",
        include_punctuation=True,
    )


def test_source_shards_are_disjoint_complete_and_stable(tmp_path) -> None:
    sources = [tmp_path / f"{index:02d}.mscx" for index in range(11)]
    shards = [
        select_source_shard(
            sources,
            shard_count=4,
            shard_index=index,
        )
        for index in range(4)
    ]
    assert [len(shard) for shard in shards] == [3, 3, 3, 2]
    assert sorted(source for shard in shards for source in shard) == sources
    assert not any(
        set(shards[left]) & set(shards[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )
    with pytest.raises(ValueError, match="shard-index"):
        select_source_shard(
            sources,
            shard_count=4,
            shard_index=4,
        )


def test_source_text_tokens_separate_lyrics_from_other_text(tmp_path) -> None:
    source = tmp_path / "score.mscx"
    source.write_text(
        """\
<museScore>
  <Text><text>Allegro con brio</text></Text>
  <Lyrics><text>Con-</text></Lyrics>
  <Lyrics><text>amore</text></Lyrics>
</museScore>
""",
        encoding="utf-8",
    )
    lyrics, other = source_text_token_counts(source)
    assert lyrics == {"con": 1, "amore": 1}
    assert other == {"allegro": 1, "con": 1, "brio": 1}
    assert text_token_keys("L’istesso  tempo.") == ("listesso", "tempo")

    remaining_lyrics = lyrics.copy()
    remaining_other = other.copy()
    assert consume_source_text_role(
        "con",
        remaining_lyrics=remaining_lyrics,
        remaining_non_lyrics=remaining_other,
    ) == "supported"
    assert consume_source_text_role(
        "Con-",
        remaining_lyrics=remaining_lyrics,
        remaining_non_lyrics=remaining_other,
    ) == "lyric"


def test_source_text_selection_excludes_music_glyph_and_out_of_scope_contexts(
    tmp_path,
) -> None:
    source = tmp_path / "contexts.mscx"
    source.write_text(
        """\
<museScore>
  <Text><style>Title</style><text>Piano Concerto 5</text></Text>
  <Tempo><text>Allegro con brio</text></Tempo>
  <StaffText><text>dolce</text></StaffText>
  <TextLine><beginText><text>cresc.</text></beginText></TextLine>
  <Glissando><text>gliss.</text></Glissando>
  <RehearsalMark><text>A</text></RehearsalMark>
  <Instrument><longName>Clarinet in B♭</longName><trackName>Editor name</trackName></Instrument>
  <Tuplet><Number><text>3</text></Number></Tuplet>
  <Note><Fingering><text>4</text></Fingering></Note>
  <FiguredBass><text>6</text></FiguredBass>
  <Harmony><text>C7</text></Harmony>
  <MeasureNumber><text>23</text></MeasureNumber>
  <Lyrics><text>sing</text></Lyrics>
</museScore>
""",
        encoding="utf-8",
    )
    included = Counter()
    excluded = Counter()
    lyrics, supported = source_text_token_counts(
        source,
        included_contexts=included,
        excluded_contexts=excluded,
    )

    assert lyrics == {"sing": 1}
    assert supported == {
        "piano": 1,
        "concerto": 1,
        "allegro": 1,
        "con": 1,
        "brio": 1,
        "dolce": 1,
        "cresc": 1,
        "gliss": 1,
        "a": 1,
        "clarinet": 1,
        "in": 1,
        "b": 1,
    }
    assert "5" not in supported
    assert excluded["Number"] == 1
    assert excluded["Fingering"] == 1
    assert excluded["FiguredBass"] == 1
    assert excluded["Harmony"] == 1
    assert excluded["MeasureNumber"] == 1


def test_ambiguous_lyric_and_direction_token_is_never_supervised() -> None:
    assert consume_source_text_role(
        "con",
        remaining_lyrics=Counter({"con": 1}),
        remaining_non_lyrics=Counter({"con": 1}),
        ambiguous_keys={"con"},
    ) == "ambiguous"


def test_source_text_tokens_read_compressed_musescore(tmp_path) -> None:
    source = tmp_path / "score.mscz"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "score.mscx",
            "<museScore><Text><text>Andante</text></Text>"
            "<Lyrics><text>sing</text></Lyrics></museScore>",
        )
        archive.writestr("META-INF/container.xml", "<container/>")

    lyrics, other = source_text_token_counts(source)

    assert lyrics == {"sing": 1}
    assert other == {"andante": 1}


def test_source_text_tokens_ignore_embedded_part_excerpts(tmp_path) -> None:
    source = tmp_path / "score-with-parts.mscz"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "score.mscx",
            "<museScore><Text><text>Full Score</text></Text></museScore>",
        )
        archive.writestr(
            "Excerpts/Violin/Violin.mscx",
            "<museScore><Text><text>Part Only</text></Text></museScore>",
        )
        archive.writestr("META-INF/container.xml", "<container/>")

    lyrics, other = source_text_token_counts(source)

    assert lyrics == {}
    assert other == {"full": 1, "score": 1}


def test_maps_pdf_top_origin_box_to_high_resolution_png() -> None:
    assert map_pdf_box_to_image(
        (10, 20, 30, 40),
        pdf_width=100,
        pdf_height=200,
        image_width=500,
        image_height=1000,
        padding_pixels=2,
    ) == (48, 98, 152, 202)


def test_maps_subpercent_pdf_png_rounding_with_independent_axes() -> None:
    assert map_pdf_box_to_image(
        (10, 20, 30, 40),
        pdf_width=100,
        pdf_height=200,
        image_width=500,
        image_height=998,
        padding_pixels=0,
    ) == pytest.approx((50, 99.8, 150, 199.6))


def test_rejects_pdf_png_aspect_mismatch() -> None:
    with pytest.raises(ValueError, match="aspect mismatch"):
        map_pdf_box_to_image(
            (0, 0, 10, 10),
            pdf_width=100,
            pdf_height=200,
            image_width=500,
            image_height=950,
        )


def test_excludes_outside_or_materially_clipped_pdf_words() -> None:
    with pytest.raises(ExcludedTextBoxError, match="invalid mapped geometry"):
        map_pdf_box_to_image(
            (10, 10, 10, 20),
            pdf_width=100,
            pdf_height=100,
            image_width=500,
            image_height=500,
        )
    with pytest.raises(ExcludedTextBoxError, match="outside rendered page"):
        map_pdf_box_to_image(
            (101, 10, 110, 20),
            pdf_width=100,
            pdf_height=100,
            image_width=500,
            image_height=500,
        )
    with pytest.raises(ExcludedTextBoxError, match="materially clipped"):
        map_pdf_box_to_image(
            (-6, 10, 20, 20),
            pdf_width=100,
            pdf_height=100,
            image_width=500,
            image_height=500,
        )
    assert map_pdf_box_to_image(
        (-1, 10, 20, 20),
        pdf_width=100,
        pdf_height=100,
        image_width=500,
        image_height=500,
    ) == (0, 48, 102, 102)

    class Page:
        width = 100
        height = 100

        @staticmethod
        def extract_words(**_kwargs):
            return [
                {
                    "text": "outside",
                    "fontname": "Edwin",
                    "size": 10,
                    "x0": 101,
                    "top": 10,
                    "x1": 110,
                    "bottom": 20,
                },
                {
                    "text": "Allegro",
                    "fontname": "Edwin",
                    "size": 10,
                    "x0": 10,
                    "top": 10,
                    "x1": 30,
                    "bottom": 20,
                },
            ]

    exclusions: Counter[str] = Counter()
    words = extract_page_words(
        Page(),
        image_width=500,
        image_height=500,
        exclusion_counts=exclusions,
    )
    assert [word["text"] for word in words] == ["Allegro"]
    assert exclusions == {"outside_page": 1}


def test_exhaustive_detection_retains_nonmusic_punctuation() -> None:
    class Page:
        width = 100
        height = 100

        @staticmethod
        def extract_words(**_kwargs):
            return [
                {
                    "text": "=",
                    "fontname": "Edwin",
                    "size": 10,
                    "x0": 10,
                    "top": 10,
                    "x1": 20,
                    "bottom": 20,
                },
                {
                    "text": "=",
                    "fontname": "Leland",
                    "size": 10,
                    "x0": 30,
                    "top": 10,
                    "x1": 40,
                    "bottom": 20,
                },
            ]

    assert extract_page_words(
        Page(),
        image_width=500,
        image_height=500,
    ) == []
    words = extract_page_words(
        Page(),
        image_width=500,
        image_height=500,
        include_nonmusic_punctuation=True,
    )
    assert [word["text"] for word in words] == ["="]


def test_resume_reuses_existing_nonempty_pdf(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "score.pdf"
    output.write_bytes(b"%PDF-existing")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("renderer must not be called for a reusable PDF")

    monkeypatch.setattr(
        "app.tools.prepare_openscore_pdf_text.subprocess.run",
        fail_run,
    )
    _render_pdf(
        tmp_path / "score.mscx",
        musescore_exe=tmp_path / "MuseScore.exe",
        output_path=output,
        timeout_seconds=30,
        reuse_existing=True,
    )
    assert output.read_bytes() == b"%PDF-existing"


def test_reuse_pdf_requires_the_signed_hash(tmp_path) -> None:
    from hashlib import sha256

    from app.tools.prepare_openscore_pdf_text import reuse_rendered_pdf

    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    source.write_bytes(b"%PDF-verified")
    expected = sha256(source.read_bytes()).hexdigest()
    assert reuse_rendered_pdf(
        source,
        output,
        expected_sha256=expected,
    )
    assert output.read_bytes() == source.read_bytes()

    source.write_bytes(b"%PDF-tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        reuse_rendered_pdf(
            source,
            tmp_path / "other.pdf",
            expected_sha256=expected,
        )
