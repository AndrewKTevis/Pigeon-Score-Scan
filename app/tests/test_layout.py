from pathlib import Path

import cv2
import numpy as np

from scorescan.layout import (
    ScoreSystemLayout,
    StaffSystem,
    _layout_boundary_warnings,
    _merge_staff_group_hypotheses,
    analyze_layout,
)


def synthetic_page(path: Path, systems: int = 3, measures: int = 4) -> None:
    image = np.full((1600, 1200), 255, dtype=np.uint8)
    for system in range(systems):
        y0 = 180 + system * 430
        spacing = 14
        for line in range(5):
            cv2.line(image, (90, y0 + line * spacing), (1110, y0 + line * spacing), 0, 2)
        for bar in range(measures + 1):
            x = 90 + round(1020 * bar / measures)
            cv2.line(image, (x, y0), (x, y0 + 4 * spacing), 0, 3)
        for note in range(9):
            x = 130 + note * 100
            y = y0 + (note % 7) * 7
            cv2.ellipse(image, (x, y), (8, 6), -20, 0, 360, 0, -1)
            cv2.line(image, (x + 7, y), (x + 7, y - 42), 0, 2)
    cv2.imwrite(str(path), image)


def test_layout_detects_staff_systems(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    synthetic_page(path)
    layout = analyze_layout(path)
    assert len(layout.systems) == 3
    assert len(layout.score_systems) == 3
    assert all(len(system.staff_indices) == 1 for system in layout.score_systems)
    assert all(system.spacing == 14 for system in layout.systems)
    assert sum(system.measure_count for system in layout.systems) >= 9
    assert layout.estimated_measure_count == sum(system.measure_count for system in layout.score_systems)


def test_page_repetitions_do_not_trigger_simultaneous_staff_limit() -> None:
    physical = [
        StaffSystem(index, [], 0, 0, 0, 0, 10.0)
        for index in range(1, 21)
    ]
    score_systems = [
        ScoreSystemLayout(
            index=index + 1,
            staff_indices=list(range(index * 4 + 1, index * 4 + 5)),
            top=0,
            bottom=0,
            left=0,
            right=0,
            spacing=10.0,
        )
        for index in range(5)
    ]

    assert _layout_boundary_warnings(physical, score_systems) == []
    oversized = ScoreSystemLayout(
        index=1,
        staff_indices=list(range(1, 18)),
        top=0,
        bottom=0,
        left=0,
        right=0,
        spacing=10.0,
    )
    assert _layout_boundary_warnings(physical, [oversized]) == [
        "单个总谱系统超过 16 行物理谱表，超出高准确度边界"
    ]


def test_relaxed_staff_groups_only_fill_missing_bands() -> None:
    primary = [
        [100.0, 110.0, 120.0, 130.0, 140.0],
        [300.0, 310.0, 320.0, 330.0, 340.0],
    ]
    supplemental = [
        # A shifted relaxed copy must not move the primary geometry.
        [101.5, 111.5, 121.5, 131.5, 141.5],
        # A complete faded staff in an otherwise empty band is admitted.
        [200.0, 210.0, 220.0, 230.0, 240.0],
        # Beam/text rows with incompatible spacing are rejected.
        [400.0, 414.0, 428.0, 442.0, 456.0],
    ]

    merged = _merge_staff_group_hypotheses(primary, supplemental)

    assert merged == [
        primary[0],
        supplemental[1],
        primary[1],
    ]


def test_relaxed_staff_group_requires_line_to_interline_contrast() -> None:
    primary = [
        [100.0, 110.0, 120.0, 130.0, 140.0],
        [300.0, 310.0, 320.0, 330.0, 340.0],
    ]
    real_faded = [200.0, 210.0, 220.0, 230.0, 240.0]
    flat_gray_fold = [400.0, 410.0, 420.0, 430.0, 440.0]
    gray = np.full((500, 200), 230, dtype=np.uint8)
    for line in real_faded:
        gray[int(line) - 1:int(line) + 2, :] = 205
    gray[390:451, :] = 185

    merged = _merge_staff_group_hypotheses(
        primary,
        [real_faded, flat_gray_fold],
        gray=gray,
    )

    assert merged == [primary[0], real_faded, primary[1]]


def test_score_system_recovers_opening_barline_missing_on_most_staves() -> None:
    from scorescan.layout import StaffSystem, infer_score_systems

    def staff(index: int, y0: int, barlines: list[int], count: int) -> StaffSystem:
        return StaffSystem(
            index=index,
            line_y=[float(y0 + offset) for offset in (0, 10, 20, 30, 40)],
            top=y0 - 20,
            bottom=y0 + 60,
            left=35 if index == 1 else 80,
            right=900,
            spacing=10.0,
            barlines=barlines,
            measure_count=count,
        )

    complete = [94, 269, 406, 484, 610, 730, 817, 900]
    physical = [
        staff(1, 100, [40, *complete], 8),
        staff(2, 180, complete[1:], 7),
        staff(3, 260, complete[2:], 5),
    ]

    score_system = infer_score_systems(physical)[0]

    assert score_system.measure_count == 7
    assert score_system.left == 94
    assert score_system.barlines == complete


def test_instrument_label_prefix_is_not_counted_as_first_measure() -> None:
    """Regression for the user's trumpet + organ scan.

    The physical staff extent starts in the instrument-name column, while the
    supported opening barline is much farther right. Counting that prefix as a
    measure shifted the first ornament of each system one measure forward.
    """

    from scorescan.layout import StaffSystem, anchor_x_to_measure, infer_score_systems

    def staff(index: int, y0: int) -> StaffSystem:
        return StaffSystem(
            index=index,
            line_y=[float(y0 + offset) for offset in (0, 8, 16, 24, 32)],
            top=y0 - 34,
            bottom=y0 + 66,
            left=34 if index == 1 else 145,
            right=901,
            spacing=8.0,
            barlines=[158, 368, 472, 577, 664, 794, 900],
            measure_count=6,
        )

    system = infer_score_systems(
        [staff(1, 194), staff(2, 279), staff(3, 371)]
    )[0]

    assert system.left == 158
    first_trill = anchor_x_to_measure(system, 301.5, 6)
    fifth_trill = anchor_x_to_measure(system, 727.0, 6)
    assert (first_trill.local_index, first_trill.method) == (0, "barline_exact")
    assert (fifth_trill.local_index, fifth_trill.method) == (4, "barline_exact")


def test_layout_recovers_antialiased_nine_staff_full_score(tmp_path: Path) -> None:
    path = tmp_path / "gray-full-score.png"
    image = np.full((1200, 1000), 230, dtype=np.uint8)
    staff_starts = (150, 230, 310, 500, 580, 660, 850, 930, 1010)
    for staff_index, y0 in enumerate(staff_starts):
        for line in range(5):
            # Alternating light-grey rows reproduce a scan where global Otsu keeps
            # only four lines in several staves.
            value = 125 if (line + staff_index) % 2 == 0 else 180
            cv2.line(image, (100, y0 + line * 8), (920, y0 + line * 8), value, 1)
        for x in (100, 300, 510, 720, 920):
            cv2.line(image, (x, y0), (x, y0 + 32), 20, 1)
        cv2.ellipse(image, (180, y0 + 16), (5, 4), 0, 0, 360, 10, -1)
    cv2.imwrite(str(path), image)

    layout = analyze_layout(path)

    assert len(layout.systems) == 9
    assert [system.staff_indices for system in layout.score_systems] == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_layout_does_not_stop_after_only_the_long_dark_staves(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed-length-scan.png"
    image = np.full((1100, 1000), 238, dtype=np.uint8)
    # Two short upper staves fall below the old 32%-of-page projection floor.
    # Two long, dark lower staves are sufficient to trigger the old early return.
    for y0 in (130, 260):
        for line in range(5):
            cv2.line(
                image,
                (90, y0 + line * 15),
                (350, y0 + line * 15),
                35,
                2,
            )
        cv2.line(image, (90, y0), (90, y0 + 60), 20, 2)
        cv2.line(image, (350, y0), (350, y0 + 60), 20, 2)
    for y0 in (650, 790):
        for line in range(5):
            cv2.line(
                image,
                (80, y0 + line * 15),
                (920, y0 + line * 15),
                0,
                2,
            )
        for x in (80, 360, 640, 920):
            cv2.line(image, (x, y0), (x, y0 + 60), 0, 2)
    cv2.imwrite(str(path), image)

    layout = analyze_layout(path)

    assert len(layout.systems) == 4
    assert all(
        abs(round(system.line_y[0]) - expected) <= 1
        for system, expected in zip(
            layout.systems,
            (130, 260, 650, 790),
            strict=True,
        )
    )


def test_layout_recovers_faded_short_final_system(
    tmp_path: Path,
) -> None:
    path = tmp_path / "faded-short-final-system.png"
    image = np.full((1200, 1000), 244, dtype=np.uint8)
    for y0 in (150, 360, 570):
        for line in range(5):
            cv2.line(
                image,
                (80, y0 + line * 14),
                (920, y0 + line * 14),
                15,
                2,
            )
    y0 = 900
    for line in range(5):
        # Two lines are badly broken and therefore lose the page-wide row
        # projection vote.  The complete system is also unusually short.
        if line in (0, 3):
            for x in range(80, 350, 34):
                cv2.line(
                    image,
                    (x, y0 + line * 14),
                    (min(x + 9, 350), y0 + line * 14),
                    25,
                    1,
                )
        else:
            cv2.line(
                image,
                (80, y0 + line * 14),
                (350, y0 + line * 14),
                35,
                1,
            )
    for x in (80, 215, 350):
        cv2.line(image, (x, y0), (x, y0 + 56), 25, 1)
    cv2.imwrite(str(path), image)

    layout = analyze_layout(path)

    assert len(layout.systems) == 4
    assert abs(layout.systems[-1].line_y[0] - y0) <= 2
    assert layout.systems[-1].right - layout.systems[-1].left < 400


def test_layout_recovers_isolated_one_line_percussion_staff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ensemble-with-percussion.png"
    image = np.full((700, 1400), 245, dtype=np.uint8)
    for y0 in (120, 450):
        for line in range(5):
            cv2.line(
                image,
                (80, y0 + line * 14),
                (1320, y0 + line * 14),
                10,
                2,
            )
    cv2.line(image, (80, 320), (1320, 320), 10, 2)
    for x in (80, 400, 720, 1040, 1320):
        cv2.line(image, (x, 316), (x, 324), 10, 2)
    cv2.imwrite(str(path), image)

    layout = analyze_layout(path)

    assert len(layout.systems) == 3
    assert [len(staff.line_y) for staff in layout.systems] == [5, 1, 5]
    assert abs(layout.systems[1].line_y[0] - 320) <= 1


def test_layout_reconstructs_faded_staff_from_two_long_line_remnants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide-ensemble-faded-interior-staff.png"
    image = np.full((760, 4000), 245, dtype=np.uint8)
    for y0 in (100, 540):
        for offset in (0, 20, 40, 60, 80):
            cv2.line(image, (80, y0 + offset), (3920, y0 + offset), 10, 2)
        for x in (80, 900, 1800, 2700, 3920):
            cv2.line(image, (x, y0), (x, y0 + 80), 10, 2)

    # The interior staff is real, but only its second and fifth lines retain
    # page-wide support. The other three lines survive in a common local span.
    for offset in (0, 40, 60):
        cv2.line(
            image,
            (1850, 300 + offset),
            (2120, 300 + offset),
            10,
            2,
        )
    for offset in (20, 80):
        cv2.line(
            image,
            (80, 300 + offset),
            (3920, 300 + offset),
            10,
            2,
        )
    for x in (1850, 1985, 2120):
        cv2.line(image, (x, 300), (x, 380), 10, 2)
    cv2.imwrite(str(path), image)

    layout = analyze_layout(path)

    assert len(layout.systems) == 3
    assert [len(staff.line_y) for staff in layout.systems] == [5, 5, 5]
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            layout.systems[1].line_y,
            (300, 320, 340, 360, 380),
            strict=True,
        )
    ) <= 2.5


def test_layout_does_not_merge_two_real_single_line_staves(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ensemble-with-two-single-line-staves.png"
    image = np.full((760, 4000), 245, dtype=np.uint8)
    for y0 in (100, 540):
        for offset in (0, 20, 40, 60, 80):
            cv2.line(image, (80, y0 + offset), (3920, y0 + offset), 10, 2)
        for x in (80, 900, 1800, 2700, 3920):
            cv2.line(image, (x, y0), (x, y0 + 80), 10, 2)
    for y in (320, 380):
        cv2.line(image, (80, y), (3920, y), 10, 2)
        for x in (80, 900, 1800, 2700, 3920):
            cv2.line(image, (x, y - 4), (x, y + 4), 10, 2)
    cv2.imwrite(str(path), image)

    layout = analyze_layout(path)

    assert [len(staff.line_y) for staff in layout.systems] == [5, 1, 1, 5]
    assert [
        round(staff.line_y[0])
        for staff in layout.systems
        if len(staff.line_y) == 1
    ] == [320, 380]


def test_infer_score_systems_groups_piano_staves_without_summing_measure_counts() -> None:
    from scorescan.layout import PageLayout, StaffSystem, infer_score_systems

    def staff(index: int, y0: int) -> StaffSystem:
        return StaffSystem(
            index=index,
            line_y=[y0 + offset for offset in (0, 10, 20, 30, 40)],
            top=y0 - 25,
            bottom=y0 + 65,
            left=80,
            right=1120,
            spacing=10,
            barlines=[80, 340, 600, 860, 1120],
            measure_count=4,
        )

    physical = [staff(1, 100), staff(2, 190), staff(3, 420), staff(4, 510)]
    score_systems = infer_score_systems(physical)
    layout = PageLayout(1200, 800, physical, 0.95, score_systems=score_systems)

    assert [system.staff_indices for system in score_systems] == [[1, 2], [3, 4]]
    assert [system.measure_count for system in score_systems] == [4, 4]
    assert score_systems[0].barlines == [80, 340, 600, 860, 1120]
    assert layout.estimated_measure_count == 8
    assert layout.to_dict()["physical_staff_count"] == 4
    assert layout.to_dict()["score_system_count"] == 2
    # Geometry groups horizontal systems only. It must not guess whether the
    # two staves are one piano part or two independent instruments.
    assert layout.part_groups == []


def test_infer_score_systems_recovers_dense_repeated_piano_topology() -> None:
    from scorescan.layout import PageLayout, StaffSystem, infer_score_systems

    line_starts = (
        328.5,
        577.5,
        945.5,
        1211.5,
        1527.0,
        1805.0,
        2146.5,
        2417.0,
        2763.0,
        3007.5,
        3311.0,
        3558.5,
        3936.0,
        4158.5,
    )
    measure_counts = (3, 3, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 1)
    physical = [
        StaffSystem(
            index=index,
            line_y=[y0 + offset for offset in (0.0, 27.0, 54.5, 81.5, 108.5)],
            top=int(y0 - 54),
            bottom=int(y0 + 162),
            left=180,
            right=3200,
            spacing=27.25,
            barlines=[180, 920, 1680, 2440, 3200],
            measure_count=measure_count,
        )
        for index, (y0, measure_count) in enumerate(
            zip(line_starts, measure_counts, strict=True),
            start=1,
        )
    ]

    score_systems = infer_score_systems(physical)
    layout = PageLayout(3395, 4628, physical, 0.95, score_systems=score_systems)

    assert [system.staff_indices for system in score_systems] == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
    ]
    assert {system.grouping_method for system in score_systems} == {
        "periodic_bimodal_vertical_gap"
    }
    assert layout.estimated_measure_count == 25


def test_infer_score_systems_rejects_smooth_nonperiodic_gap_clusters() -> None:
    from scorescan.layout import StaffSystem, infer_score_systems

    starts = [100]
    for gap_ratio in (5.0, 6.0, 7.0, 8.0, 9.0):
        starts.append(starts[-1] + 40 + int(gap_ratio * 10))
    physical = [
        StaffSystem(
            index=index,
            line_y=[y0 + offset for offset in (0, 10, 20, 30, 40)],
            top=y0 - 20,
            bottom=y0 + 60,
            left=80,
            right=1120,
            spacing=10,
            barlines=[80, 340, 600, 860, 1120],
            measure_count=4,
        )
        for index, y0 in enumerate(starts, start=1)
    ]

    score_systems = infer_score_systems(physical)

    assert len(score_systems) == 1
    assert score_systems[0].staff_indices == [1, 2, 3, 4, 5, 6]
    assert score_systems[0].grouping_method == "absolute_vertical_gap"


def test_candidate_conditioned_layout_distinguishes_repeated_solo_and_piano_systems() -> None:
    from scorescan.layout import PageLayout, StaffSystem

    physical = [
        StaffSystem(
            index=index,
            line_y=[y0 + offset for offset in (0, 10, 20, 30, 40)],
            top=y0 - 20,
            bottom=y0 + 60,
            left=80,
            right=1120,
            spacing=10,
            barlines=[80, 340, 600, 860, 1120],
            measure_count=5,
        )
        for index, y0 in enumerate((100, 190, 310, 400, 520, 610), start=1)
    ]
    layout = PageLayout(1200, 800, physical, 0.95)

    solo = layout.expectation_for_staff_topology(1)
    piano = layout.expectation_for_staff_topology(2)
    six_staff_score = layout.expectation_for_staff_topology(6)

    assert (solo.score_system_count, solo.measure_count) == (6, 30)
    assert (piano.score_system_count, piano.measure_count) == (3, 15)
    assert (six_staff_score.score_system_count, six_staff_score.measure_count) == (1, 5)
    assert solo.incomplete_staff_count == 0
    assert layout.to_dict()["measure_count_hypotheses"] == [5, 10, 15, 30]


def test_anchor_x_to_measure_uses_irregular_barline_widths() -> None:
    from scorescan.layout import StaffSystem, anchor_x_to_measure

    system = StaffSystem(
        index=1,
        line_y=[100, 112, 124, 136, 148],
        top=60,
        bottom=190,
        left=100,
        right=1000,
        spacing=12.0,
        barlines=[250, 600, 780],
        measure_count=4,
    )
    anchor = anchor_x_to_measure(system, 500, 4)
    assert anchor.local_index == 1
    assert 0.65 < anchor.offset_ratio < 0.75
    assert anchor.method == "barline_exact"
    assert anchor.confidence > 0.90


def test_anchor_x_to_measure_rescales_small_count_mismatch() -> None:
    from scorescan.layout import StaffSystem, anchor_x_to_measure

    system = StaffSystem(
        index=1,
        line_y=[100, 112, 124, 136, 148],
        top=60,
        bottom=190,
        left=100,
        right=1000,
        spacing=12.0,
        barlines=[300, 700],
        measure_count=3,
    )
    anchor = anchor_x_to_measure(system, 520, 4)
    assert 0 <= anchor.local_index < 4
    assert 0.0 <= anchor.offset_ratio < 1.0
    assert anchor.method == "barline_rescaled"
    assert 0.40 <= anchor.confidence < 0.92
