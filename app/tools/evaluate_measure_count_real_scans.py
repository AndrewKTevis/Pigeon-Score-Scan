from __future__ import annotations

"""Maintained real-page regression for measure-count evidence fusion.

When scan images are available, page geometry is extracted directly. Release rebuilds
may instead replay the committed layout measurements from an earlier report; the report
records that provenance explicitly. OMR count observations remain constructed from the
manually verified totals because release builders do not bundle homr model weights.
This is a layout/count regression, not an end-to-end OMR benchmark.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from scorescan.layout import analyze_layout
from scorescan.measure_count_resolver import MeasureCountResolver


@dataclass(frozen=True)
class Candidate:
    variant: str
    measure_count: int
    valid: bool = True
    agreement_ratio: float = 0.90
    calibrated_probability: float = 0.80
    raw_score: float = 980.0
    measure_gap_penalty: float = 0.0


def _candidate_sets() -> dict[str, tuple[Candidate, ...]]:
    return {
        "allegretto": (
            Candidate("primary", 46),
            Candidate("flat", 46),
            Candidate("otsu", 45, agreement_ratio=0.46, calibrated_probability=0.40, raw_score=850.0),
        ),
        "bwv147": (
            Candidate("primary", 44),
            Candidate("flat", 44),
            Candidate("otsu", 44),
            Candidate("adaptive", 45, agreement_ratio=0.48, calibrated_probability=0.36, raw_score=840.0),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allegretto", type=Path)
    parser.add_argument("--bwv", type=Path)
    parser.add_argument("--replay-layout-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truths = {"allegretto": 46, "bwv147": 44}
    paths = {"allegretto": args.allegretto, "bwv147": args.bwv}
    replay: dict[str, dict[str, object]] = {}
    provenance = "scan_images"
    if args.replay_layout_report is not None:
        payload = json.loads(args.replay_layout_report.read_text(encoding="utf-8"))
        replay = {
            str(item.get("name")): item
            for item in payload.get("cases", [])
            if isinstance(item, dict)
        }
        provenance = f"replayed_layout_measurements:{args.replay_layout_report.name}"
    elif not all(path is not None for path in paths.values()):
        parser.error("provide both scan images or --replay-layout-report")

    resolver = MeasureCountResolver()
    candidates = _candidate_sets()
    results = []
    for name in ("allegretto", "bwv147"):
        if replay:
            source = replay.get(name)
            if not source:
                raise ValueError(f"missing maintained layout case: {name}")
            layout_count = int(source["layout_count"])
            layout_confidence = float(source["layout_confidence"])
        else:
            path = paths[name]
            assert path is not None
            layout = analyze_layout(path)
            layout_count = sum(system.measure_count for system in layout.systems)
            layout_confidence = layout.confidence
        resolution = resolver.resolve(
            layout_count=layout_count,
            layout_confidence=layout_confidence,
            candidates=candidates[name],
        )
        results.append(
            {
                "name": name,
                "truth": truths[name],
                "layout_count": layout_count,
                "layout_confidence": layout_confidence,
                "resolved_count": resolution.selected_count,
                "correct": resolution.selected_count == truths[name],
                "source": resolution.source,
                "probability": resolution.probability,
                "margin": resolution.margin,
            }
        )
    output = {
        "model_version": resolver.model_version,
        "layout_provenance": provenance,
        "cases": results,
        "accuracy": sum(item["correct"] for item in results) / max(len(results), 1),
        "scope": "maintained real-scan layout measurements plus constructed OMR count evidence; not end-to-end OMR",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
