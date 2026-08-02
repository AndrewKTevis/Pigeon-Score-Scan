# Changelog

This project follows [Semantic Versioning](https://semver.org/) after the first stable release.

## Unreleased

- No changes yet.

## 0.38.0 — 2026-08-03

- Bundled Python, locked dependencies and recognition models in the Windows package.
- Removed first-run and first-conversion downloads from the product runtime.
- Added an output reminder to assign the correct MuseScore instrument to each part.

## 0.37.0 — 2026-08-02

- Published the first stable Windows release and signed source tag.
- Updated ONNX Runtime, the development toolchain and pinned GitHub Actions.
- Kept OpenCV on the latest homr-compatible 4.x release.
- Added deterministic source and Windows archives with SHA-256 verification.

## 0.37.0-dev — 2026-08-02

- Expanded the score model for common piano notation, vertically stacked monophonic ensembles and piano-plus-ensemble scores.
- Added multi-page MXL/MusicXML export, local preview, resumable jobs and artifact integrity checks.
- Added bounded source-backed checks for pitch, rhythm, ties, slurs, hairpins, text directions, articulations and ornaments.
- Replaced browser-only startup with a Windows desktop shell and tray lifecycle.
- Removed automatic page rotation and CUDA runtime support from the product path.
