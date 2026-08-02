# Scope and limitations

This document defines the 0.37 recognition boundary. Expanding the boundary requires new tests and independent real-scan evidence.

## Input contract

- Printed Western staff notation from a scan or a clean scan-derived image.
- Pages must have the correct orientation before import.
- Mild scanning noise, uneven illumination and ordinary page curvature are accepted.
- Perspective photographs, severe occlusion, handwriting and arbitrary screenshots are outside the high-accuracy contract.

## Supported score structures

- Solo single-staff instrumental music.
- Common piano and keyboard notation.
- One keyboard part may use up to four physical staves, including temporary extra staves and ossias.
- A keyboard staff may contain independent simultaneous voices, voice crossing and cross-staff notes, beams, slurs and ties.
- Vertically stacked monophonic instrumental parts.
- Piano plus vertically stacked monophonic instrumental parts.
- Up to 16 physical staves per score system.

Each ensemble part has its own semantic timeline. Horizontal note position does not force simultaneity between instruments.

## Page handling

Multiple scans imported in one task are treated as consecutive pages of one score and exported as one MXL/MusicXML document. PDF pages are expanded in document order. Independently scanned instrumental parts are not merged into a reconstructed full score.

## Notation target

The pipeline handles the ordinary printed notation used by the supported score structures, including clefs, key and time signatures, notes, rests, chords, tuplets, accidentals, ties, slurs, beams, dynamics, hairpins, common articulations, ornaments, tempo text, rehearsal text and repeat/barline structures.

The output preserves musical semantics; exact source engraving, font choice and line breaking are not guaranteed.

## Excluded notation

- Percussion and unpitched notation
- Tablature
- Lyrics
- Chord symbols and figured bass
- Condensed or divisi staves whose independent timelines cannot be represented by the declared staff contract
- True polymeter
- Microtonal and graphic notation
- Handwritten music

## Release claims

Passing structural checks does not prove that every symbol matches the source. Stable accuracy claims require a work-disjoint frozen real-scan benchmark with published per-configuration metrics. Until those gates pass, the application remains a development release and may require manual review.
