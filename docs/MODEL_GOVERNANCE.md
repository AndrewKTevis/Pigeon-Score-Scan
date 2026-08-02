# Model governance

Pigeon Score Scan combines deterministic notation checks with bounded CPU models. The following rules apply to every committed model resource.

## Required metadata

- Semantic role and authority level
- Ordered feature contract
- Training seed and code entry point
- Grouped train, calibration and validation splits
- Threshold-selection rule
- Independent confirmation metrics
- Resource size and SHA-256

`app/src/scorescan/resources/model_manifest.json` is the runtime source of truth. CI regenerates the manifest and rejects an unexpected diff.

## Data separation

Works, pages and derived crops remain grouped. A work assigned to training cannot appear in calibration, validation or the frozen release benchmark. Test results may not guide feature selection, model selection or threshold choice.

User scans are never training data unless the user supplies separate, explicit permission and the material's copyright permits that use. The public repository contains no training corpus or third-party checkpoint without verified redistribution rights.

## Deployment rule

Models do not receive unrestricted MusicXML authority. Deterministic code defines the eligible transaction and verifies the result. A model may accept, reject or rank only within its declared feature and topology boundary.

Synthetic accuracy, component accuracy and end-to-end real-scan accuracy are reported separately.
