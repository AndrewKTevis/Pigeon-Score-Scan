# Architecture

## Runtime

`launcher.zig` starts the portable Windows runtime. The Python service binds to loopback, creates a per-launch access token and serves the desktop WebView. Closing the desktop shell either returns control to the tray or terminates the service, depending on the selected action.

The supported runtime is CPU-only. OMR workers run as subprocess trees so cancellation and shutdown do not leave detached work behind.

## Recognition pipeline

1. Transactional image/PDF import and deterministic page ordering.
2. Scan normalization without automatic rotation.
3. Staff-system, barline and page-layout analysis.
4. Isolated whole-page recognition candidates from bounded preprocessing families.
5. Measure alignment and conversion to immutable Score IR.
6. Family-balanced semantic consensus.
7. Narrow source-backed transactions for supported pitch, rhythm and notation relations.
8. MusicXML merge, pagination, preview and artifact verification.

Ensemble parts keep independent timelines. Score-system and measure geometry may align parts, but note x-coordinates are not treated as rhythmic evidence across instruments.

## Model authority

Bundled models are CPU resources recorded in `model_manifest.json` with role, feature order, version, size and SHA-256. They may rank or veto an already bounded proposal. They cannot create an independent evidence family, bypass XML or rhythm validation, or directly commit arbitrary MusicXML.

Missing or damaged resources fail conservatively. Model and deterministic transaction behavior is covered by focused contract tests.

## Persistence

Each task has a versioned job record, immutable inputs, resumable checkpoints and a verified result bundle. A process lock prevents two application instances from mutating the same workspace. Downloads and file-opening actions resolve only through the task root.

## Release boundary

Portable packages contain the application, runtime bootstrap scripts, locked dependencies and required CPU resources. They exclude tests, training data, task workspaces, local caches, diagnostics and developer tools. Source and portable archives are generated deterministically by `app/tools/build_release.py`.
