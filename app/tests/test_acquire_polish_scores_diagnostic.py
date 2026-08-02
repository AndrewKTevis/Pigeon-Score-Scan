from __future__ import annotations

from app.tools.acquire_polish_scores_diagnostic import (
    musicxml_work_fingerprint,
)


def _xml(title: str, note: str) -> bytes:
    return f"""<?xml version="1.0"?>
<score-partwise version="4.0">
  <work><work-title>{title}</work-title></work>
  <identification><creator type="composer">Composer</creator></identification>
  <part-list><score-part id="P1"><part-name>Solo</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><note><pitch><step>{note}</step>
  <octave>4</octave></pitch><duration>1</duration></note></measure></part>
</score-partwise>""".encode()


def test_work_fingerprint_groups_pages_by_score_metadata() -> None:
    assert musicxml_work_fingerprint(_xml("Work A", "C")) == (
        musicxml_work_fingerprint(_xml("Work A", "D"))
    )
    assert musicxml_work_fingerprint(_xml("Work A", "C")) != (
        musicxml_work_fingerprint(_xml("Work B", "C"))
    )


def test_work_fingerprint_groups_missing_metadata_conservatively() -> None:
    first = b"<score-partwise><part-list/></score-partwise>"
    second = b"<score-partwise><part-list><score-part/></part-list></score-partwise>"
    assert musicxml_work_fingerprint(first) == musicxml_work_fingerprint(second)
