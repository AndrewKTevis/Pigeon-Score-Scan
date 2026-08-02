# PaddleOCR training patches

This directory contains the small PaddleOCR source subset exercised by Pigeon
Score Scan's training-contract tests. The files derive from PaddleOCR revision
`2661c7c0` and retain its Apache License 2.0 terms.

The changes add bounded epoch chunks, preserve resume metadata, and restrict
hard-negative crops to explicitly authorized pages. They are development-only
training inputs and are not imported by the desktop runtime.

The complete upstream project is available at
<https://github.com/PaddlePaddle/PaddleOCR>.
