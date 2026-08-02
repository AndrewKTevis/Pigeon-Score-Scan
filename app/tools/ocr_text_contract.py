"""Shared, lightweight contract for source-proven score text datasets."""

from __future__ import annotations


SOURCE_TEXT_SELECTION_VERSION = "source-proven-supported-score-text-only@1"

# Text stored under other score objects (for example Tuplet, Fingering,
# FiguredBass, Harmony, or Lyrics) must never become OCR supervision.
SUPPORTED_TEXT_OBJECTS = frozenset(
    {
        "Dynamic",
        "Expression",
        "Glissando",
        "InstrumentChange",
        "Jump",
        "Marker",
        "PlayTechAnnotation",
        "RehearsalMark",
        "StaffText",
        "SystemText",
        "Tempo",
        "Text",
        "beginText",
        "continueText",
        "endText",
    }
)

# MuseScore stores printed full/abbreviated instrument labels outside a
# nested <text> element. trackName is only an editor label and is omitted.
SUPPORTED_DIRECT_TEXT_TAGS = frozenset({"longName", "shortName"})
