# Pigeon Score Scan

[![CI](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/ci.yml/badge.svg)](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/ci.yml)
[![CodeQL](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/codeql.yml/badge.svg)](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/codeql.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

[中文](README.zh-CN.md)

Pigeon Score Scan is a local Windows desktop application that converts correctly oriented scans of printed Western staff notation into MXL and MusicXML. Recognition runs on the CPU. Files remain on the computer running the application.

> **Project status:** 0.37 development snapshot. The source and test suite are public, but no stable accuracy guarantee or production release is claimed yet.

## Supported material

- Single-staff instrumental parts.
- Common piano notation, including independent voices on a keyboard staff and cross-staff notation.
- Vertically stacked monophonic ensemble parts. Instruments do not need note-by-note horizontal alignment.
- Piano combined with monophonic instrumental parts.
- Correctly oriented image scans and PDFs, with consecutive pages imported in score order.

The current verification target allows up to 16 physical staves per system. A keyboard part may use temporary extra staves or ossias. See [Scope and limitations](docs/SCOPE.md) for the complete contract.

Not supported: handwritten music, photographs with perspective distortion, tablature, percussion notation, lyrics, chord symbols, figured bass, condensed/divisi staves, true polymeter, microtonal or graphic notation, and reconstruction of separately scanned instrumental parts into one score.

## Output

- MXL, intended for opening and editing in MuseScore.
- Uncompressed MusicXML.
- A local preview and a machine-readable conversion report.

The application does not silently release a result as verified. Structural and integrity checks are reported with the output when manual review is required.

## Privacy and security

The desktop service binds to `127.0.0.1` and uses an access token for local requests. Imported scans, intermediate files and results stay in the selected portable workspace. Do not expose the service port to a network.

Security reports should use [GitHub private vulnerability reporting](SECURITY.md). Do not attach copyrighted scores unless you have permission to share them.

## Development

Requirements:

- Windows 10 or Windows 11
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/) 0.9.26
- Node.js for the JavaScript syntax check
- Zig 0.16.0 only when rebuilding the Windows launcher

PowerShell setup:

```powershell
git clone https://github.com/KalePotato/Pigeon-Score-Scan.git
Set-Location Pigeon-Score-Scan

py -3.12 -m pip install uv==0.9.26
uv sync --project app --locked --group dev
$env:PYTHONPATH = (Resolve-Path "app/src").Path

uv run --project app python -m scorescan --self-test --json --root .
uv run --project app python -m pytest -q app/tests
node --check app/src/scorescan/web/app.js
```

Start the development application:

```powershell
$env:PYTHONPATH = (Resolve-Path "app/src").Path
uv run --project app python -m scorescan
```

The portable Windows package is built by the release process, not committed to the source repository. See [Building and testing](BUILDING.md).

## Repository layout

```text
app/src/scorescan/   application and recognition pipeline
app/tests/           automated tests
app/tools/           build, evaluation and model-training tools
training/            reproducibility metadata and bounded orchestration scripts
runtime/             portable bootstrap scripts
licenses/            third-party license texts
third_party/         reviewed upstream training patches
docs/                scope, architecture and release policy
launcher.zig         Windows desktop launcher source
```

The 2026-08-02 Windows verification run for this public tree completed 1,382 tests. Ten corpus-integration tests were skipped because their optional external datasets are not distributed in the repository. CI reruns the same public suite from the committed source and lock file.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. The project is licensed under [GNU AGPL v3 or later](LICENSE). Third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and `licenses/`.
