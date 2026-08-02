from scorescan.text_enrichment import _assemble_phrase_rows


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_adjacent_music_words_form_phrase() -> None:
    rows = [
        ("Allegro", .91, box(10, 10, 95, 35), "tesseract"),
        ("con", .88, box(105, 10, 140, 35), "tesseract"),
        ("brio", .90, box(150, 10, 195, 35), "tesseract"),
        ("mf", .94, box(30, 90, 55, 115), "rapid"),
    ]
    merged = _assemble_phrase_rows(rows)
    assert any(item[0] == "Allegro con brio" for item in merged)
    assert any(item[0] == "mf" for item in merged)


def test_distant_words_are_not_joined() -> None:
    rows = [
        ("Allegro", .91, box(10, 10, 95, 35), "tesseract"),
        ("dolce", .90, box(500, 10, 560, 35), "tesseract"),
    ]
    merged = _assemble_phrase_rows(rows)
    assert len(merged) == 2


def test_staff_internal_fragments_are_not_joined() -> None:
    from scorescan.layout import PageLayout, StaffSystem

    system = StaffSystem(
        index=1,
        line_y=[100, 112, 124, 136, 148],
        top=55,
        bottom=190,
        left=0,
        right=600,
        spacing=12.0,
        barlines=[200, 400],
        measure_count=3,
    )
    layout = PageLayout(600, 240, [system], 0.98)
    rows = [
        ("Pot", .76, box(120, 112, 165, 138), "tesseract"),
        ("i", .72, box(170, 112, 180, 138), "tesseract"),
        ("Ù", .74, box(185, 112, 205, 138), "tesseract"),
    ]
    merged = _assemble_phrase_rows(rows, layout)
    assert [item[0] for item in merged] == ["Pot", "i", "Ù"]


def test_phrase_join_does_not_degrade_strong_single_word() -> None:
    rows = [
        ("Allegretto", .93, box(10, 10, 115, 35), "tesseract"),
        (">", .60, box(120, 10, 132, 35), "tesseract"),
        ("deel", .55, box(138, 10, 180, 35), "tesseract"),
    ]
    merged = _assemble_phrase_rows(rows)
    assert any(item[0] == "Allegretto" for item in merged)
    assert not any(item[0] == "Allegretto > deel" for item in merged)
