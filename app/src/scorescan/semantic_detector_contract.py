"""Dependency-free release and runtime contracts for the semantic detector."""

SEMANTIC_DETECTOR_MANIFEST_NAME = "semantic_detector.json"
SEMANTIC_DETECTOR_FORMAT = 1
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CATEGORIES_BYTES = 1024 * 1024
MAX_ONNX_BYTES = 512 * 1024 * 1024
MINIMUM_INDEPENDENT_WORKS = 200
MINIMUM_OPERATING_POINT_PRECISION = 0.995
MINIMUM_OPERATING_POINT_RECALL = 0.98
MINIMUM_OPERATING_POINT_TRUE_POSITIVES = 25
MINIMUM_HIGH_RECALL_MARK_RECALL = 0.99
FIXED_RARE_CLASS_OPERATING_POINT_THRESHOLD = 0.995
FIXED_RARE_CLASS_SELECTION_METHOD = "fixed_contract_rare_class"
CALIBRATED_OPERATING_POINT_SELECTION_METHOD = "development_calibrated"
SEMANTIC_PAGE_NMS_IOU = 0.75
SEMANTIC_DETECTOR_INPUT_SIZE = 1024
SEMANTIC_DETECTOR_TARGET_STAFF_SPACING = 21.0
SEMANTIC_DETECTOR_TILE_OVERLAP = 256
SEMANTIC_DETECTOR_MAXIMUM_TILES = 96
SEMANTIC_DETECTOR_PAGE_SHAPE_CONTRACT = (
    "ordinary-single-page-or-two-page-spread-aspect-ratio@1"
)
SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO = 3.0
SEMANTIC_DETECTOR_MINIMUM_SCALE = 0.5
SEMANTIC_DETECTOR_MAXIMUM_SCALE = 2.0
TILE_FRAGMENT_FUSION_VERSION = (
    "one-to-one-opposing-tile-boundary-min-confidence-structural-fragment-fusion@3"
)
GEOMETRY_CORROBORATION_CLASSES = frozenset({"hairpin", "slur", "tie"})
TEXT_REGION_CLASSES = frozenset(
    {
        "expressionText",
        "fingeringText",
        "genericDynamic",
        "instrumentNameText",
        "jumpText",
        "markerText",
        "measureNumberText",
        "rehearsalMarkText",
        "scoreText",
        "staffText",
        "systemText",
        "techniqueText",
        "tempoText",
    }
)
TILE_FRAGMENT_FUSION_CLASSES = frozenset(
    {
        "beam",
        "bracket",
        "genericBarline",
        "glissando",
        "hairpin",
        "ottava",
        "pedal",
        "slur",
        "textLine",
        "tie",
        "trillExtension",
        "tuplet",
        "volta",
    }
) | TEXT_REGION_CLASSES
TILE_FRAGMENT_STAFF_AGNOSTIC_CLASSES = frozenset(
    {"bracket", "genericBarline"}
)
if not TILE_FRAGMENT_STAFF_AGNOSTIC_CLASSES <= TILE_FRAGMENT_FUSION_CLASSES:
    raise RuntimeError("staff-agnostic fragments must be fusion classes")
# These regions are useful OCR proposals, but they are not generic timeline
# directions.  Their source role must remain authoritative: instrument names
# belong to score metadata, measure numbers are structural labels, fingerings
# require note-level ownership, and rehearsal marks require dedicated
# MusicXML semantics.  Until the corresponding dedicated writer has enough
# source evidence, treating any of them as ``direction-type/words`` creates a
# plausible-looking but semantically misplaced result.
NON_DIRECTION_TEXT_REGION_CLASSES = frozenset(
    {
        "fingeringText",
        "instrumentNameText",
        "measureNumberText",
        "rehearsalMarkText",
    }
)
if not NON_DIRECTION_TEXT_REGION_CLASSES < TEXT_REGION_CLASSES:
    raise RuntimeError("non-direction text classes must be a strict text-region subset")
SYMBOL_AUDIT_CLASSES = frozenset(
    {
        "arpeggio",
        "augmentationDot",
        "beam",
        "bracket",
        "breathMark",
        "fermata",
        "flag",
        "genericAccidental",
        "genericArticulation",
        "genericBarline",
        "genericClef",
        "genericKeySignature",
        "genericOrnament",
        "genericRest",
        "genericTimeSignature",
        "glissando",
        "graceSlash",
        "ottava",
        "parenthesis",
        "pedal",
        "textLine",
        "tremoloBetweenNotes",
        "tremoloSingle",
        "trillExtension",
        "tuplet",
        "volta",
    }
)
# These classes directly address the product's highest-cost unattended-output
# failures: false accidentals, missing/extra relations, dynamics, articulations,
# ornaments and hairpins. Their frozen independent holdout therefore needs a
# stronger recall floor than the already strict all-class operating point.
HIGH_RECALL_MARK_CLASSES = frozenset(
    {
        "genericAccidental",
        "genericArticulation",
        "genericDynamic",
        "genericOrnament",
        "hairpin",
        "slur",
        "tie",
    }
)
# Only accidental inventory is used bidirectionally to flag both omissions and
# extras. Other high-recall classes still provide omission/corroboration
# evidence, because a detector miss alone cannot prove an emitted relation or
# expressive mark is wrong.
POSITIONAL_INVENTORY_CLASSES = frozenset({"genericAccidental"})


def page_aspect_ratio(width: int, height: int) -> float:
    """Return an orientation-independent page aspect ratio.

    ScoreScan accepts ordinary portrait/landscape scan pages and two-page
    spreads. Whole-work vertical scrolls and horizontal panoramas must be split
    into physical pages before recognition.
    """

    if width <= 0 or height <= 0:
        raise ValueError("scan page dimensions must be positive")
    return max(width / height, height / width)


def page_shape_is_supported(width: int, height: int) -> bool:
    return (
        page_aspect_ratio(width, height)
        <= SEMANTIC_DETECTOR_MAXIMUM_PAGE_ASPECT_RATIO
    )


SUPPORTED_RUNTIME_CLASSES = (
    GEOMETRY_CORROBORATION_CLASSES
    | TEXT_REGION_CLASSES
    | SYMBOL_AUDIT_CLASSES
)
if not TILE_FRAGMENT_FUSION_CLASSES <= SUPPORTED_RUNTIME_CLASSES:
    raise RuntimeError("tile-fragment fusion classes must be runtime classes")
