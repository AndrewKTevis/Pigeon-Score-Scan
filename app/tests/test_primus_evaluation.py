from pathlib import Path

from scorescan.primus_evaluation import (
    aggregate_primus_reports,
    compare_primus_semantics,
    parse_musicxml_semantics,
    parse_primus_semantic_text,
)


def test_primus_semantic_and_musicxml_equivalence(tmp_path: Path) -> None:
    reference = parse_primus_semantic_text(
        "clef-G2 keySignature-EbM timeSignature-3/4 "
        "note-Bb5_quarter note-Eb5_eighth. rest-sixteenth barline"
    )
    candidate_path = tmp_path / "candidate.musicxml"
    candidate_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1"><measure number="1">
    <attributes><divisions>4</divisions></attributes>
    <attributes>
      <key><fifths>-3</fifths></key>
      <time><beats>3</beats><beat-type>4</beat-type></time>
      <clef><sign>G</sign><line>2</line></clef>
    </attributes>
    <note><pitch><step>B</step><alter>-1</alter><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note>
    <note><pitch><step>E</step><alter>-1</alter><octave>5</octave></pitch><duration>3</duration><type>eighth</type><dot/></note>
    <note><rest/><duration>1</duration><type>16th</type></note>
  </measure></part>
</score-partwise>""",
        encoding="utf-8",
    )

    report = compare_primus_semantics(reference, parse_musicxml_semantics(candidate_path))

    assert report["semantic_exact"]
    assert report["event_error_rate"] == 0.0
    assert report["header_accuracy"] == 1.0


def test_primus_evaluation_detects_pitch_and_rhythm_errors(tmp_path: Path) -> None:
    reference = parse_primus_semantic_text(
        "clef-G2 timeSignature-C note-C4_quarter barline"
    )
    candidate_path = tmp_path / "candidate.musicxml"
    candidate_path.write_text(
        """<score-partwise version="4.0">
<part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes>
<time><beats>4</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>D</step><octave>4</octave></pitch><type>half</type></note>
</measure></part></score-partwise>""",
        encoding="utf-8",
    )

    report = compare_primus_semantics(reference, parse_musicxml_semantics(candidate_path))

    assert not report["semantic_exact"]
    assert report["pitch_accuracy"] == 0.0
    assert report["rhythm_accuracy"] == 0.0
    assert report["event_error_rate"] == 1.0


def test_primus_aggregate_reports_micro_event_error_rate() -> None:
    aggregate = aggregate_primus_reports(
        [
            {"reference_event_count": 2, "event_edit_count": 1, "event_error_rate": 0.5,
             "event_presence_precision": 1.0, "event_presence_recall": 0.5,
             "event_kind_accuracy": 0.5, "pitch_accuracy": 1.0, "rhythm_accuracy": 1.0,
             "fermata_accuracy": 1.0, "header_accuracy": 1.0, "semantic_exact": False,
             "measure_count_exact": True, "clef_exact": True, "key_signature_exact": True,
             "time_signature_exact": True, "tie_count_exact": True},
            {"reference_event_count": 8, "event_edit_count": 1, "event_error_rate": 0.125,
             "event_presence_precision": 1.0, "event_presence_recall": 0.875,
             "event_kind_accuracy": 0.875, "pitch_accuracy": 1.0, "rhythm_accuracy": 1.0,
             "fermata_accuracy": 1.0, "header_accuracy": 1.0, "semantic_exact": False,
             "measure_count_exact": True, "clef_exact": True, "key_signature_exact": True,
             "time_signature_exact": True, "tie_count_exact": True},
        ]
    )

    assert aggregate["total_reference_event_count"] == 10
    assert aggregate["total_event_edit_count"] == 2
    assert aggregate["micro_event_error_rate"] == 0.2
