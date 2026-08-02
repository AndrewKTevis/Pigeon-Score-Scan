from scorescan.text_enrichment import _merge_ocr_rows


def box(x1, y1, x2, y2):
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


def test_ocr_consensus_merges_overlapping_passes() -> None:
    rows = [
        ("Allegro con brlo", .74, box(10,10,200,40), "rapid-original"),
        ("Allegro con brio", .69, box(12,9,202,41), "rapid-no-lines"),
        ("Allegro con brio", .72, box(11,11,201,42), "tesseract-no-lines"),
        ("mf", .88, box(30,100,60,130), "rapid-original"),
    ]
    merged = _merge_ocr_rows(rows)
    assert len(merged) == 2
    allegro = next(item for item in merged if item[0].startswith("Allegro"))
    assert allegro[0] == "Allegro con brio"
    assert allegro[1] > .72
