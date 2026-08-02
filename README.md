# Pigeon Score Scan

[![CI](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/ci.yml/badge.svg)](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/ci.yml)
[![CodeQL](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/codeql.yml/badge.svg)](https://github.com/KalePotato/Pigeon-Score-Scan/actions/workflows/codeql.yml)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

[中文](README.zh-CN.md)

Pigeon Score Scan is a local Windows application for converting printed staff-notation scans to MXL and MusicXML. Recognition runs on the CPU; scans and results remain on the computer running the application.

## Download

Download the Windows x64 package from [Releases](https://github.com/KalePotato/Pigeon-Score-Scan/releases), extract the complete ZIP, and run `pigeon-score-scan.exe`.

The first launch requires an internet connection to install the pinned Python runtime, dependencies and recognition models. Later launches use the local portable runtime. Current packages are unsigned development releases, so Windows may display a security warning.

## Supported scores

- Single-staff instrumental parts.
- Common piano notation, including independent voices and cross-staff notation.
- Vertically stacked monophonic ensemble parts without note-by-note horizontal alignment.
- Piano combined with monophonic instrumental parts.
- Correctly oriented image scans and PDFs, imported in score order.

The verified boundary allows up to 16 physical staves per system. Handwritten music, perspective-distorted photographs, percussion notation, tablature, lyrics, chord symbols, figured bass, condensed or divisi staves, true polymeter, microtonal notation and graphic notation are outside the current scope. See [Scope and limitations](docs/SCOPE.md).

## Output

The application exports MXL for use in MuseScore, uncompressed MusicXML, a local preview and a conversion report. Results that fail an output check are marked for manual review.

## Development

Windows 10 or 11 and Python 3.12 or 3.13 are supported. Setup, test and release commands are in [BUILDING.md](BUILDING.md). Contributions are described in [CONTRIBUTING.md](CONTRIBUTING.md).

Security reports should use [private vulnerability reporting](SECURITY.md).

## License

Pigeon Score Scan is licensed under [GNU AGPL v3 or later](LICENSE). Third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and `licenses/`.
