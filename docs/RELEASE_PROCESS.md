# Release process

1. Freeze the commit and version.
2. Run CI, CodeQL and the checks in `PUBLIC_RELEASE_CHECKLIST.md`.
3. Run the physical Windows 10/11 matrix from clean machines.
4. Build the launcher with the pinned Zig version.
5. Build deterministic source and portable archives.
6. Inspect the portable file list and run its self-test.
7. Calculate SHA-256 after the final build.
8. Create a signed tag from the verified commit.
9. Attach archives and checksums to the GitHub release.

Release notes state the supported score boundary, known limitations, benchmark evidence and any migration requirement. A failed verification gate blocks the release; it is not converted into advisory wording.
