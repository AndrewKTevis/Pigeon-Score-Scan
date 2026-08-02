from __future__ import annotations

"""Audit Audiveris strictly as a first-time-signature oracle.

The report intentionally ignores every note, rest, direction and layout item in the
Audiveris export.  Inputs are copied to unique diagnostic names so one batched JVM can
process them without basename collisions.  This tool is diagnostic only; it does not
authorize a product edit or provide production benchmark evidence.
"""

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_time(payload: str | bytes) -> tuple[int, int, str] | None:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", "replace")
    try:
        node = ElementTree.fromstring(payload).find(".//time")
        if node is None:
            return None
        return (
            int(node.findtext("beats") or "0"),
            int(node.findtext("beat-type") or "0"),
            str(node.get("symbol") or "").strip().casefold(),
        )
    except (ElementTree.ParseError, TypeError, ValueError):
        return None


def _read_mxl(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        candidates = sorted(
            name
            for name in archive.namelist()
            if name.casefold().endswith((".xml", ".musicxml"))
            and "meta-inf" not in name.casefold()
        )
        if not candidates:
            raise ValueError("MXL contains no score XML")
        return archive.read(candidates[0])


def _expected_rows(
    prepared_root: Path,
    image_root: Path,
    splits: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in splits:
        source = prepared_root / f"{split}.pickle"
        for item in pickle.loads(source.read_bytes()):
            expected = _first_time(item.get("musicxml") or item.get("lmx") or "")
            if expected is None or expected[1] != 2:
                continue
            relative = Path(str(item["path"]) + ".png")
            image = image_root / relative
            document = relative.parts[1] if len(relative.parts) > 1 else "unknown"
            diagnostic_name = f"{split}__{document}__{relative.stem}.png"
            rows.append(
                {
                    "split": split,
                    "source_path": str(image.resolve()),
                    "source_sha256": _sha256(image),
                    "diagnostic_name": diagnostic_name,
                    "expected": list(expected),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.resolve()
    inputs = output / "inputs"
    exports = output / "exports"
    inputs.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    rows = _expected_rows(
        args.prepared_root.resolve(),
        args.image_root.resolve(),
        tuple(args.splits),
    )
    input_paths: list[Path] = []
    for row in rows:
        destination = inputs / str(row["diagnostic_name"])
        shutil.copy2(Path(str(row["source_path"])), destination)
        input_paths.append(destination)

    command = [
        str(args.audiveris.resolve()),
        "-batch",
        "-export",
        "-output",
        str(exports),
        "--",
        *(str(path) for path in input_paths),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout_seconds,
        check=False,
    )
    (output / "audiveris.log").write_text(completed.stdout or "", encoding="utf-8")

    false_cut_count = 0
    exact_count = 0
    available_count = 0
    cut_count = 0
    cut_exact_count = 0
    numeric_count = 0
    numeric_exact_count = 0
    for row in rows:
        expected = tuple(row["expected"])
        is_cut = expected == (2, 2, "cut")
        cut_count += int(is_cut)
        numeric_count += int(not is_cut)
        export = exports / (Path(str(row["diagnostic_name"])).stem + ".mxl")
        row["export_path"] = str(export)
        if not export.is_file():
            row["status"] = "export_missing"
            row["predicted"] = None
            row["exact"] = False
            continue
        try:
            predicted = _first_time(_read_mxl(export))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            row["status"] = f"export_invalid:{type(exc).__name__}"
            row["predicted"] = None
            row["exact"] = False
            continue
        row["export_sha256"] = _sha256(export)
        row["predicted"] = list(predicted) if predicted is not None else None
        row["status"] = "available" if predicted is not None else "time_missing"
        if predicted is None:
            row["exact"] = False
            continue
        available_count += 1
        exact = predicted[:2] == expected[:2]
        row["exact"] = exact
        exact_count += int(exact)
        cut_exact_count += int(is_cut and exact)
        numeric_exact_count += int(not is_cut and exact)
        false_cut_count += int(not is_cut and predicted[:2] == (2, 2))

    return {
        "schema": "scorescan-audiveris-time-signature-diagnostic@1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "diagnostic_only": True,
        "production_evidence_eligible": False,
        "audiveris_path": str(args.audiveris.resolve()),
        "audiveris_sha256": _sha256(args.audiveris.resolve()),
        "returncode": completed.returncode,
        "requested_count": len(rows),
        "available_count": available_count,
        "exact_count": exact_count,
        "exact_rate_available": exact_count / available_count if available_count else 0.0,
        "cut_count": cut_count,
        "cut_exact_count": cut_exact_count,
        "cut_recall": cut_exact_count / cut_count if cut_count else 0.0,
        "numeric_count": numeric_count,
        "numeric_exact_count": numeric_exact_count,
        "numeric_exact_rate": numeric_exact_count / numeric_count if numeric_count else 0.0,
        "numeric_false_cut_count": false_cut_count,
        "safe_for_product_integration": bool(
            completed.returncode == 0
            and available_count == len(rows)
            and false_cut_count == 0
            and exact_count == len(rows)
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audiveris", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "calibration", "candidate_test"),
        default=("train", "calibration", "candidate_test"),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    report = run(args)
    destination = args.output.resolve() / "report.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
