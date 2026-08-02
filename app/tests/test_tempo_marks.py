from scorescan.tempo_marks import parse_metronome_mark
from scorescan.text_enrichment import classify_text


def test_unicode_and_word_metronome_marks() -> None:
    mark = parse_metronome_mark("♩. = ca. 72")
    assert mark is not None
    assert mark.beat_unit == "quarter"
    assert mark.dotted
    assert mark.per_minute_low == 72
    assert mark.approximate

    ranged = parse_metronome_mark("M.M. quarter = 120–126")
    assert ranged is not None
    assert ranged.per_minute_high == 126
    assert classify_text("𝅘𝅥𝅮 = 132") == "metronome"


def test_metronome_parser_rejects_ambiguous_numbers() -> None:
    assert parse_metronome_mark("Op. 72") is None
    assert parse_metronome_mark("= 88") is None
    assert parse_metronome_mark("♩ = 900") is None


def test_metronome_parser_accepts_common_ocr_quarter_note_substitution() -> None:
    mark = parse_metronome_mark("d=62")
    assert mark is not None
    assert mark.beat_unit == "quarter"
    assert mark.per_minute_low == 62
