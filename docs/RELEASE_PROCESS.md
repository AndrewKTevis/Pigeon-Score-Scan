# Release process

1. Freeze the commit and version.
2. Run CI, CodeQL and the checks in `PUBLIC_RELEASE_CHECKLIST.md`.
3. Run the physical Windows 10/11 matrix from clean machines.
4. Build the launcher with the pinned Zig version.
5. Prepare and verify the pinned offline Python, dependency and model runtime.
6. Build deterministic source and portable archives.
7. Inspect the portable file list, confirm that it contains no download bootstrap, and run its self-test with external connections blocked.
8. Calculate SHA-256 after the final build.
9. Create a signed tag from the verified commit.
10. Attach archives and checksums to the GitHub release.

Release notes state the supported score boundary, known limitations, available benchmark evidence and any migration requirement. A failed engineering verification gate blocks the release. Missing recognition evidence blocks the corresponding accuracy claim and must be disclosed.
