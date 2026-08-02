from __future__ import annotations

from scorescan.barline_sequence import BarlineSequenceClassifier, refine_barline_sequence
from scorescan.layout import PageLayout, StaffSystem


def test_sequence_refiner_removes_low_confidence_false_split() -> None:
    # Regression shape from the supplied Allegretto scan.  The first candidate splits
    # one plausible opening measure into two short intervals and is weaker than both
    # its edge and right-hand neighbour.
    candidates = [
        (0.849867, 202),
        (0.922263, 324),
        (0.935015, 539),
        (0.920533, 754),
        (0.918464, 964),
        (0.881051, 1148),
        (0.992147, 1318),
    ]
    refined = refine_barline_sequence(left=66, right=1322, spacing=11.5, candidates=candidates)
    retained = [item.x for item in refined if item.retained]
    assert BarlineSequenceClassifier().enabled
    assert 202 not in retained
    assert retained == [324, 539, 754, 964, 1148, 1318]


def test_sequence_refiner_preserves_pickup_measure_boundary() -> None:
    candidates = [(0.94, 110), (0.93, 315), (0.95, 520), (0.91, 730), (0.96, 940)]
    refined = refine_barline_sequence(left=40, right=1000, spacing=12.0, candidates=candidates)
    assert [item.x for item in refined if item.retained] == [110, 315, 520, 730, 940]


def test_sequence_refiner_does_not_downweight_system_edge_barline() -> None:
    candidates = [(0.91, 260), (0.94, 480), (0.97, 698)]
    refined = refine_barline_sequence(left=40, right=700, spacing=12.0, candidates=candidates)
    final = refined[-1]
    assert final.retained
    assert final.final_probability == final.local_probability


def test_layout_round_trip_preserves_sequence_confidence() -> None:
    layout = PageLayout(
        1000,
        1400,
        [
            StaffSystem(
                1,
                [100, 112, 124, 136, 148],
                60,
                190,
                30,
                970,
                12.0,
                [250, 500],
                3,
                [0.98, 0.87],
                [0.96, 0.91],
            )
        ],
        0.95,
    )
    restored = PageLayout.from_dict(layout.to_dict())
    assert restored.systems[0].barline_sequence_confidences == [0.96, 0.91]


def test_sequence_refiner_iteratively_removes_separated_false_stems() -> None:
    # Two false strokes are visible in different measures.  Recomputing neighbourhoods
    # after each deletion must remove both while preserving every labelled boundary.
    candidates = [
        (0.778106187366397, 140),
        (0.8928785873462488, 212),
        (0.9777213357482804, 336),
        (0.9238627958227947, 472),
        (0.8154158211882507, 665),
        (0.9600084974084311, 798),
        (0.6339330232234691, 963),
        (0.7986739154605194, 1115),
        (0.7657950963671702, 1163),
        (0.8921480171948549, 1267),
    ]
    refined = refine_barline_sequence(
        left=42,
        right=1365,
        spacing=11.627293750096781,
        candidates=candidates,
    )
    removed = [item.x for item in refined if not item.retained]
    assert removed == [140, 1163]


def test_sequence_refiner_uses_neutral_fallback_when_model_is_disabled() -> None:
    class DisabledClassifier:
        enabled = False

        def predict(self, _features: object) -> float:
            raise AssertionError("disabled model must not be evaluated")

    candidates = [(0.72, 150), (0.91, 300), (0.93, 500)]
    refined = refine_barline_sequence(
        left=40,
        right=700,
        spacing=12.0,
        candidates=candidates,
        classifier=DisabledClassifier(),  # type: ignore[arg-type]
    )
    assert all(item.retained for item in refined)
    assert all(item.sequence_probability == 0.5 for item in refined)
