# Security policy

## Supported versions

Security fixes are applied to the newest published release and the current `main` branch. Development snapshots may change without compatibility guarantees.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/AndrewKTevis/Pigeon-Score-Scan/security/advisories/new). Do not open a public issue before a fix is available.

Include:

- Pigeon Score Scan version or commit;
- Windows version;
- reproduction steps and expected impact;
- a privacy-safe diagnostics bundle, if relevant.

Do not attach access tokens, user workspaces or copyrighted scores.

## Security boundary

- The service must bind only to `127.0.0.1`.
- Local API requests require a per-launch access token.
- Uploaded and generated paths are confined to the active task root.
- Model resources are checked against a committed SHA-256 manifest.
- Image evidence is size- and format-checked before decoding.
- Recognition workers run in isolated subprocess trees.
- Only one process may own a mutable portable workspace.

The browser service is not designed for network deployment. Exposing its port invalidates the supported security boundary.
