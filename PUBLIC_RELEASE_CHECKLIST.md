# Public release checklist

## Source

- [ ] Version is consistent in `VERSION`, `app/pyproject.toml` and release notes.
- [ ] `uv lock --project app --check` passes.
- [ ] The public-tree audit reports no local paths, secrets, generated binaries or oversized files.
- [ ] The model manifest regenerates without a diff.
- [ ] Third-party notices and bundled licenses match the distributed dependencies.

## Verification

- [ ] Full Windows test suite passes from a clean checkout.
- [ ] Portable self-test has zero critical failures.
- [ ] The Windows archive contains the pinned offline Python, dependencies and recognition models and contains no download bootstrap.
- [ ] Startup and model initialization succeed while external socket connections are blocked.
- [ ] Windows 10 and 11 startup, tray, Unicode paths and paths containing spaces are exercised.
- [ ] Multi-page image and PDF inputs produce one ordered score.
- [ ] MXL opens in the current MuseScore release.
- [ ] Interrupted conversion resumes without corrupting the workspace.
- [ ] Antivirus and SmartScreen behavior is documented for the unsigned or signed build.

## Recognition evidence

- [ ] The frozen real-scan benchmark is work-disjoint and matches the declared scope.
- [ ] Per-configuration metrics and failure classes are published.
- [ ] No training, threshold-selection or validation leakage is present.
- [ ] Accuracy wording matches the measured evidence and does not imply zero-error recognition.

## Artifacts

- [ ] Source and portable archives are deterministic.
- [ ] SHA-256 files are generated after the final build.
- [ ] Release artifacts contain no workspace, cache, diagnostics, local paths or training data.
- [ ] The release tag is signed and CI completes on the exact tagged commit.
