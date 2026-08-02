from scorescan.direction_model import DirectionCorrector
from scorescan.text_enrichment import classify_text


def test_direction_corrector() -> None:
    corrector = DirectionCorrector()
    assert corrector.model_enabled
    assert corrector.model_format == 3
    assert corrector.model_version == "scorescan-direction-logistic-9"
    result = corrector.suggest("Allegro con brlo")
    assert result.text == "Allegro con brio"
    assert result.probability > 0.9


def test_musical_text_classification() -> None:
    assert classify_text("Allegro con brio") == "direction"
    assert classify_text("rit.") == "direction"
    assert classify_text("mf") == "dynamic"
    assert classify_text("quarter = 120") == "metronome"


def test_direction_corrector_compositional_phrases() -> None:
    corrector = DirectionCorrector()
    safe_cases = {
        "conbrio": "con brio",
        "Maestoso esostenuto": "Maestoso e sostenuto",
        "pocoa poco e dim.": "poco a poco e dim.",
        "And ante": "Andante",
    }
    for observed, expected in safe_cases.items():
        result = corrector.suggest(observed)
        assert result.text == expected
        assert corrector.should_autocorrect(result)

    # The suggestion is useful, but changing two independent OCR tokens is not safe
    # enough for unattended MusicXML write-back. It must remain a review item.
    uncertain = corrector.suggest("rall a fempo")
    assert uncertain.text == "rall. a tempo"
    assert not corrector.should_autocorrect(uncertain)


def test_direction_corrector_preserves_exact_allargando() -> None:
    corrector = DirectionCorrector()
    exact = corrector.suggest("allargando")
    assert exact.text == "allargando"
    assert not exact.changed

    for observed in ("allargand a", "a llargando"):
        result = corrector.suggest(observed)
        assert result.text == "allargando"
        # A one-character split may represent a separate musical token, so it is
        # offered for review rather than written back silently.
        assert not corrector.should_autocorrect(result)


def test_direction_corrector_preserves_terminal_period_on_known_tempo() -> None:
    corrector = DirectionCorrector()
    result = corrector.suggest("Allegretto.")
    assert result.text == "Allegretto."
    assert not result.changed
    assert result.method == "terminal-punctuation-preserve"
    assert result.safety_class == "source-punctuation"
    assert result.probability >= 0.90

    # Internal and repeated punctuation are not silently blessed by this narrow rule.
    assert corrector.suggest("Andante dolce. ma spressivo").method != "terminal-punctuation-preserve"
    assert corrector.suggest("Allegretto..").method != "terminal-punctuation-preserve"


def test_direction_corrector_never_silently_invents_missing_prefix() -> None:
    corrector = DirectionCorrector()
    for observed in ("a Capo", "t iempo", "subitio"):
        result = corrector.suggest(observed)
        assert not corrector.should_autocorrect(result)


def test_direction_corrector_preserves_understood_novel_composition() -> None:
    corrector = DirectionCorrector()
    for observed in ("Andante dolce ma espressivo", "andante dolce ma espressivo"):
        result = corrector.suggest(observed)
        assert result.text == observed
        assert result.method == "token-preserve"
        assert not result.changed
        assert not corrector.should_autocorrect(result)


def test_direction_corrector_does_not_invent_digits_or_trust_unsafe_punctuation() -> None:
    corrector = DirectionCorrector()
    for observed in ("Langsam vnd gesangvoll", "With warmth knd expression"):
        result = corrector.suggest(observed)
        assert "2nd" not in result.text.casefold()
        assert not corrector.should_autocorrect(result)

    punctuated = corrector.suggest("Andante dolce. ma spressivo")
    assert punctuated.text == "Andante dolce. ma espressivo"
    assert not corrector.should_autocorrect(punctuated)

    singleton = corrector.suggest("Moderato con calma c semplicità")
    assert not corrector.should_autocorrect(singleton)


def test_invalid_direction_model_falls_back_without_raising(tmp_path) -> None:
    import json

    from scorescan.direction_model import DIRECTION_FEATURE_NAMES

    model_path = tmp_path / "invalid_direction_model.json"
    model_path.write_text(
        json.dumps(
            {
                "format": 3,
                "model_version": "invalid-test",
                "feature_names": list(DIRECTION_FEATURE_NAMES),
                "intercept": 0.0,
                "coefficients": [0.0] * len(DIRECTION_FEATURE_NAMES),
                "means": [0.0] * len(DIRECTION_FEATURE_NAMES),
                "scales": [1.0] * (len(DIRECTION_FEATURE_NAMES) - 1) + [0.0],
                "lexicon": [{"text": "Allegro", "frequency": 1.0}],
            }
        ),
        encoding="utf-8",
    )

    corrector = DirectionCorrector(model_path, verify_manifest=False)
    assert not corrector.model_enabled
    result = corrector.suggest("Allegro con brlo")
    assert result.text == "Allegro con brlo"
    assert result.method == "disabled"
    assert not result.changed
    assert not corrector.should_autocorrect(result)
