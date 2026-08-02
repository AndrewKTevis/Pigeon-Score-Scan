from pathlib import Path

from lxml import etree

from scorescan.score_ir import audit_score, measure_distance, score_from_tree


def parse(xml: str) -> etree._ElementTree:
    return etree.ElementTree(etree.fromstring(xml.encode()))


def test_score_ir_normalizes_different_divisions() -> None:
    left = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note></measure></part></score-partwise>''')
    right = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>quarter</type></note></measure></part></score-partwise>''')
    a = score_from_tree(left).measures[0]
    b = score_from_tree(right).measures[0]
    assert measure_distance(a, b) == 0.0


def test_score_ir_pitch_difference_is_detected() -> None:
    left = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>1</divisions></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note></measure></part></score-partwise>''')
    right = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>1</divisions></attributes><note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note></measure></part></score-partwise>''')
    distance = measure_distance(score_from_tree(left).measures[0], score_from_tree(right).measures[0])
    assert 0.15 < distance < 0.6


def test_score_ir_ignores_out_of_scope_semantic_lyrics_for_product_identity() -> None:
    without_lyrics = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>1</divisions></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note></measure></part></score-partwise>''')
    with_lyrics = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>1</divisions></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type><lyric><syllabic>single</syllabic><text>la</text></lyric></note></measure></part></score-partwise>''')
    left = score_from_tree(without_lyrics).measures[0]
    right = score_from_tree(with_lyrics).measures[0]
    assert right.notes[0].lyrics == (("la", "single", ""),)
    assert left.fingerprint == right.fingerprint
    assert measure_distance(left, right) == 0.0


def test_audit_flags_type_duration_mismatch_and_multiple_voices() -> None:
    tree = parse('''<score-partwise><part id="P1"><measure number="1"><attributes><divisions>4</divisions></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>8</duration><voice>1</voice><type>quarter</type></note><backup><duration>8</duration></backup><note><pitch><step>E</step><octave>4</octave></pitch><duration>8</duration><voice>2</voice><type>half</type></note></measure></part></score-partwise>''')
    issues = audit_score(score_from_tree(tree))
    codes = {item.code for item in issues}
    assert "type_duration_mismatch" in codes
    assert "multiple_voices" in codes
