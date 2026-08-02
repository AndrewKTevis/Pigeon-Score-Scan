# Contributing

Pigeon Score Scan accepts focused fixes, tests, documentation and bounded recognition improvements.

## Before opening a pull request

1. Open an issue for changes that alter the product boundary, output semantics, model authority or persisted job format.
2. Keep changes within the scope described in [docs/SCOPE.md](docs/SCOPE.md).
3. Add a regression test for every bug fix and a contract test for every new invariant.
4. Run the checks in [BUILDING.md](BUILDING.md).
5. Keep commits limited to one logical change. Do not mix formatting rewrites with functional work.

## Recognition and model changes

- Deterministic MusicXML, rhythm, voice, topology and integrity checks remain authoritative.
- Learned components may rank or veto bounded proposals; they may not manufacture independent evidence or bypass structural validation.
- Record the model role, feature order, seed, grouped data split, calibration rule and resource SHA-256.
- Training and validation works must be disjoint. Test data may not influence training, threshold selection or model selection.
- Regenerate `model_manifest.json` and include focused parity tests.
- Accuracy claims must state the dataset, split, sample count and metric. Synthetic results are not real-scan production evidence.

## Data and privacy

Never commit user scans, filenames, task workspaces, model checkpoints without redistribution rights, or datasets whose license is unclear. Use minimal synthetic fixtures where possible. Remove personal paths and machine identifiers from logs before attaching them to an issue.

## Pull request review

A pull request must pass CI, preserve the declared scope and include enough evidence for another contributor to reproduce the result. Maintainers may request a smaller patch when review boundaries are unclear.
