from __future__ import annotations

from lxml import etree

from scorescan.score_ir import measure_distance, score_from_tree


def _score(lyric: str) -> bytes:
    lyric_xml = f"<lyric><syllabic>single</syllabic><text>{lyric}</text></lyric>" if lyric else ""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1"><measure number="1">
    <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
    <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type>{lyric_xml}</note>
  </measure></part>
</score-partwise>'''.encode()


def test_out_of_scope_lyric_difference_does_not_steer_measure_distance() -> None:
    left = score_from_tree(etree.ElementTree(etree.fromstring(_score("la")))).measures[0]
    right = score_from_tree(etree.ElementTree(etree.fromstring(_score("do")))).measures[0]
    same = score_from_tree(etree.ElementTree(etree.fromstring(_score("la")))).measures[0]

    assert measure_distance(left, same) == 0.0
    assert measure_distance(left, right) == 0.0
