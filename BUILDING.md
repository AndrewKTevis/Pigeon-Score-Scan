# Building and testing

## Reproducible Python environment

The application targets Python 3.12 and 3.13. Dependencies are fixed in `app/uv.lock`.

```powershell
py -3.12 -m pip install uv==0.9.26
uv lock --project app --check
uv sync --project app --locked --group dev
$env:PYTHONPATH = (Resolve-Path "app/src").Path
```

Pigeon Score Scan intentionally installs `opencv-python-headless`. Installing a second OpenCV wheel into the same environment is unsupported.

## Required checks

```powershell
uv run --project app python app/tools/check_public_tree.py .
uv run --project app python -m compileall -q app/src app/tools app/tests
node --check app/src/scorescan/web/app.js
uv run --project app python -m pytest -q app/tests
uv run --project app python -m scorescan --self-test --json --root .
```

Regenerate the resource manifest and require a clean diff:

```powershell
uv run --project app python app/tools/generate_model_manifest.py `
  --resources app/src/scorescan/resources
git diff --exit-code -- app/src/scorescan/resources/model_manifest.json
```

## Windows launcher

The launcher is built from `launcher.zig`. The published compiler contract is Zig 0.16.0 for `x86_64-windows`.

```powershell
zig fmt --check launcher.zig
zig build-exe launcher.zig -O ReleaseSmall -target x86_64-windows -fstrip --subsystem windows
```

Do not commit the compiled executable.

## Deterministic source archive

```powershell
uv run --project app python app/tools/build_release.py `
  --source-root . `
  --output-dir dist `
  --version 0.37.0-dev
```

Portable Windows packages require the verified launcher and pinned `uv.exe`. Release artifacts are attached to a signed GitHub release and are never committed to `main`.
