from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scorescan.direction_model import DIRECTION_FEATURE_NAMES, DirectionCorrector, _WORD_RE, feature_vector, normalize_direction  # noqa: E402

BASE_TEMPI = [
    "Larghissimo", "Grave", "Largo", "Larghetto", "Lento", "Adagio", "Adagietto",
    "Andante", "Andantino", "Marcia moderato", "Andante moderato", "Moderato",
    "Allegretto", "Allegro moderato", "Allegro", "Vivace", "Vivacissimo", "Allegrissimo",
    "Presto", "Prestissimo",
]
TEMPO_MODIFIERS = [
    "assai", "molto", "ma non troppo", "non troppo", "con moto", "con brio", "con fuoco",
    "con spirito", "con anima", "grazioso", "cantabile", "espressivo", "maestoso",
    "sostenuto", "tranquillo", "energico", "deciso", "leggiero", "dolce", "semplice",
    "appassionato", "agitato", "scherzando", "misterioso", "brillante", "alla marcia",
]
DIRECTIONS = [
    "a tempo", "Tempo I", "Tempo primo", "L'istesso tempo", "istesso tempo", "rubato",
    "rit.", "ritardando", "ritenuto", "rall.", "rallentando", "allargando", "calando",
    "smorzando", "morendo", "stringendo", "accel.", "accelerando", "più mosso",
    "meno mosso", "poco a poco", "subito", "sempre", "senza rit.", "a piacere",
    "ad lib.", "ad libitum", "dolce", "dolcissimo", "cantabile", "cantando",
    "espressivo", "con espressione", "teneramente", "con grazia", "con energia",
    "con slancio", "con dolore", "con tenerezza", "marcato", "legato", "staccato",
    "sostenuto", "tranquillo", "animato", "vivo", "comodo", "deciso", "energico",
    "maestoso", "grazioso", "leggiero", "brillante", "agitato", "appassionato",
    "misterioso", "scherzando", "semplice", "con sordino", "senza sordino",
    "sul ponticello", "sul tasto", "ordinario", "arco", "pizzicato", "pizz.",
    "solo", "tutti", "divisi", "unisono", "Fine", "D.C. al Fine", "D.S. al Coda",
]
DYNAMICS = [
    "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff", "fff", "ffff", "fp", "pf",
    "sf", "sfp", "sfpp", "sfz", "sffz", "rf", "rfz", "fz", "subito p", "subito f",
    "p dolce", "f marcato", "cresc.", "crescendo", "dim.", "diminuendo", "decresc.",
]
METRONOME = [
    "quarter = 60", "quarter = 72", "quarter = 84", "quarter = 96", "quarter = 108",
    "quarter = 120", "dotted quarter = 60", "dotted quarter = 72", "eighth = 120",
    "half = 60", "M.M. = 80", "ca. 72", "circa 96", "quarter = 120-126",
    "♩ = 60", "♩ = 72", "♩ = 84", "♩ = 96", "♩ = 108", "♩ = 120",
    "♩. = 60", "♩. = 72", "♪ = 120", "𝅘𝅥 = 88", "𝅘𝅥. = 66", "𝅘𝅥𝅮 = 132",
    "M.M. ♩ = 80", "♩ = ca. 88", "♩ = 120-126",
]


MULTILINGUAL_DIRECTIONS = [
    # French
    "Très lent", "Modéré", "Vif", "Animé", "Retenu", "En dehors", "Cédez",
    "Revenez au mouvement", "Un peu plus vite", "Un peu moins vite", "Sans presser",
    "Doux et expressif", "Avec expression", "Très chanté", "Lentement", "Vivement",
    "En animant", "En retenant", "Au mouvement", "Même mouvement",
    # German
    "Langsam", "Mäßig", "Lebhaft", "Schnell", "Sehr langsam", "Mit Ausdruck",
    "Ruhig", "Bewegt", "Nicht schleppen", "Etwas langsamer", "Etwas schneller",
    "Im Tempo", "Zart", "Kräftig", "Breit", "Innig", "Gesangvoll", "Zurückhaltend",
    # English
    "Slowly", "Moderately", "Quickly", "With expression", "Tenderly", "Broadly",
    "Freely", "In strict time", "A little faster", "A little slower", "Return to tempo",
    "Gradually faster", "Gradually slower", "With warmth", "Singing", "Flowing",
    # Spanish and general modern-language directions
    "Con expresión", "Con gracia", "Con energía", "Muy lento", "Más rápido",
    "Menos rápido", "A tiempo", "Poco a poco", "Libremente",
]

NAVIGATION = [
    "D.C.", "D.C. al Fine", "D.C. al Coda", "Da Capo", "D.S.", "D.S. al Fine",
    "D.S. al Coda", "Dal Segno", "To Coda", "al Coda", "Coda", "Segno", "Fine",
    "1st time", "2nd time", "first time", "second time",
]

EXTENDED_DIRECTIONS = [
    "Tempo giusto", "Tempo ordinario", "Tempo di marcia", "Tempo di minuetto",
    "Alla breve", "Alla polacca", "Alla siciliana", "Alla tedesca", "Alla turca",
    "Con delicatezza", "Con eleganza", "Con sentimento", "Con passione",
    "Con moto ma tranquillo", "Con molto espressione", "Con tutta forza",
    "Delicatamente", "Espressivo e cantabile", "Grazioso e leggiero",
    "Largamente", "Maestoso e sostenuto", "Marcato il canto", "Molto espressivo",
    "Non legato", "Poco meno mosso", "Poco più mosso", "Sempre legato",
    "Sempre più forte", "Sempre più piano", "Sotto voce", "Tempo rubato",
    "Un poco animato", "Un poco sostenuto", "Quasi andante", "Quasi allegretto",
    "Sans ralentir", "Très expressif", "En pressant", "En élargissant",
    "Mit innigem Ausdruck", "Sehr ausdrucksvoll", "Immer ruhiger", "Immer bewegter",
    "With great expression", "Not too fast", "Broadly and sustained", "Gently flowing",
    "cantabile ed espressivo", "dolce ma espressivo", "poco rit.", "molto rit.",
    "poco rall.", "molto rall.", "poco accel.", "senza rall.", "senza vibrato",
    "con vibrato", "sul G", "sul D", "sul A", "sul E", "harmonics", "flautando",
]


ADDITIONAL_DIRECTIONS = [
    "come prima", "come sopra", "al segno", "al fine", "senza misura",
    "senza espressione", "con affetto", "con forza", "con impeto", "con nobiltà",
    "con malinconia", "con dolcezza", "con semplicità", "con calma", "con libertà",
    "nobilmente", "grandioso", "religioso", "lamentoso", "mesto", "giocoso",
    "risoluto", "soave", "serioso", "perdendosi", "perdendo", "incalzando",
    "stretto", "largamente", "largando", "pressante", "scorrevole", "quasi niente",
    "morendo al niente", "sempre cantabile", "sempre sostenuto", "sempre espressivo",
    "poco meno", "poco più", "molto meno", "molto più", "tempo precedente",
    "au mouvement précédent", "sans ralentissement", "avec chaleur", "avec douceur",
    "très doux", "très calme", "de plus en plus animé", "de plus en plus lent",
    "Im vorigen Zeitmaß", "Mit Wärme", "Mit Ruhe", "Sehr zart", "Immer langsamer",
    "Immer schneller", "Nicht zu schnell", "Nicht zu langsam", "Gesangvoll und innig",
    "Warmly", "Calmly", "With tenderness", "With noble expression", "Without slowing",
    "A tempo subito", "Tempo I subito", "rit. poco a poco", "rall. poco a poco",
    "accel. poco a poco", "cresc. poco a poco", "dim. poco a poco",
]

HIGH_VALUE_SCAN_DIRECTIONS = [
    "Allegro con spirito", "Allegro con anima", "Allegro energico",
    "Andante espressivo", "Andante molto cantabile", "Moderato con moto",
    "Lento con espressione", "Adagio sostenuto", "Poco animato",
    "Tempo di valse", "Tempo di gavotta", "Tempo di mazurka",
    "rit. e dim.", "rall. e morendo", "cresc. ed animando",
    "poco a poco cresc.", "poco a poco dim.", "sempre più mosso",
    "sempre meno mosso", "a tempo ma tranquillo", "Tempo I ma sostenuto",
    "dolce e cantabile", "espressivo e sostenuto", "leggiero e scherzando",
    "con moto e grazia", "marcato e deciso", "cantabile con espressione",
    "M.M. ♩ = ca. 72", "M.M. ♪ = 120", "♩ = 108-112",
]

RARE_AND_EDITORIAL_DIRECTIONS = [
    "Adagio ma non troppo", "Allegro assai", "Allegro giusto", "Allegro vivace",
    "Andante con moto", "Andante sostenuto", "Lento assai", "Moderato assai",
    "Presto agitato", "Vivace ma non troppo", "Poco allegretto", "Un poco meno mosso",
    "Un poco più mosso", "Tempo I.", "Tempo 1", "Tempo 1°", "Tempo del principio",
    "in tempo", "senza tempo", "riten.", "ritenuto", "ritard.", "rallent.",
    "acceler.", "poco stringendo", "molto allargando", "cedendo", "cedez",
    "au mouvt.", "mouvement de valse", "très modéré", "assez vif", "sans lenteur",
    "en dehors et bien chanté", "mit Empfindung", "mit Leidenschaft",
    "etwas bewegt", "immer im Tempo", "sehr getragen", "nicht eilen",
    "with feeling", "with breadth", "with restrained expression", "not hurried",
    "gently but flowing", "quietly expressive", "freely, with expression",
    "pp subito", "ff subito", "sotto voce", "mezza voce", "una corda", "tre corde",
    "niente", "al niente", "dal niente", "cresc. molto", "dim. molto",
    "sempre cresc.", "sempre dim.", "poco cresc.", "poco dim.",
    "M.M. quarter = 72", "M.M. dotted quarter = 60", "quarter = ca. 88",
    "dotted eighth = 96", "half note = 54", "quarter note = 100",
    "Allegro ma non troppo e molto espressivo", "Andante sostenuto e cantabile",
    "Poco più mosso con anima", "Poco meno mosso e tranquillo",
    "Très calme et très expressif", "Mit innigem und warmem Ausdruck",
    "Gently flowing, with expression", "Broadly, but not slower",
]

COMPOSITIONAL_THRESHOLD = [
    "Allegro molto cantabile e sostenuto",
    "Andante dolce ma espressivo",
    "Moderato con calma e semplicità",
    "Sempre più dolce e cantabile",
    "Poco meno mosso ma sostenuto",
    "Lento molto espressivo e cantabile",
    "Allegretto con grazia e leggiero",
    "Adagio dolce e tranquillo",
    "Très lent et expressif",
    "Un peu plus vite et animé",
    "En retenant et très expressif",
    "Avec chaleur et douceur",
    "Langsam und gesangvoll",
    "Ruhig und sehr zart",
    "Nicht zu schnell und ruhig",
    "Mit Wärme und Ausdruck",
    "With warmth and expression",
    "Gently flowing and sustained",
    "A little faster and singing",
    "Broadly and with feeling",
    "Con calma e dolcezza",
    "Poco animato e leggiero",
    "Sempre sostenuto e tranquillo",
    "Molto espressivo ma semplice",
]

COMPOSITIONAL_TEST = [
    "Allegro con anima ma non troppo",
    "Andante molto dolce e cantabile",
    "Moderato sostenuto e tranquillo",
    "Sempre più mosso ma leggiero",
    "Adagio con calma e dolcezza",
    "Vivace con spirito e brillante",
    "Larghetto maestoso e sostenuto",
    "Poco più mosso e cantabile",
    "Très calme et chanté",
    "Un peu moins vite et très doux",
    "En animant et en retenant",
    "Avec expression et chaleur",
    "Sehr langsam und innig",
    "Lebhaft und kräftig",
    "Etwas schneller und bewegt",
    "Mit Ausdruck und Wärme",
    "Tenderly and with expression",
    "Slowly but freely",
    "With tenderness and warmth",
    "A little slower but flowing",
    "Sempre cantabile e dolce",
    "Poco meno mosso e dolce",
    "Molto sostenuto ma cantabile",
    "Con energia e brillante",
]

# Backward-compatible import used by older development evaluators.
COMPOSITIONAL_EVAL = COMPOSITIONAL_TEST


CONFUSIONS = {
    "l": ["1", "i"], "i": ["l", "1"], "1": ["l", "i"], "o": ["0", "a"], "0": ["o"],
    "m": ["rn", "n"], "n": ["m", "r"], "r": ["n"], "f": ["t"], "t": ["f"],
    "c": ["e"], "e": ["c"], "u": ["v"], "v": ["u"], "s": ["5"], "z": ["2"],
    ".": [",", ""], "'": ["", "’"], " ": ["  ", ""],
}


def build_corpus() -> list[tuple[str, float]]:
    counter: Counter[str] = Counter()
    for term in BASE_TEMPI + DIRECTIONS + DYNAMICS + METRONOME + MULTILINGUAL_DIRECTIONS + NAVIGATION + EXTENDED_DIRECTIONS + ADDITIONAL_DIRECTIONS + RARE_AND_EDITORIAL_DIRECTIONS + HIGH_VALUE_SCAN_DIRECTIONS:
        counter[term] += 8
    for base in BASE_TEMPI:
        for modifier in TEMPO_MODIFIERS:
            counter[f"{base} {modifier}"] += 3
    for prefix in ["poco", "molto", "sempre", "subito", "più", "meno"]:
        for term in ["dolce", "espressivo", "cantabile", "marcato", "legato", "animato"]:
            counter[f"{prefix} {term}"] += 2
    # Compound directions commonly printed on one line.
    for left in ["rit.", "rall.", "cresc.", "dim.", "poco a poco"]:
        for right in ["e dim.", "e cresc.", "a tempo", "molto", "sempre"]:
            counter[f"{left} {right}"] += 1
    return sorted(counter.items(), key=lambda item: normalize_direction(item[0]))


def corrupt(text: str, rng: random.Random, severity: int | None = None) -> str:
    chars = list(text)
    operations = severity if severity is not None else rng.choices([0, 1, 2, 3], [0.12, 0.48, 0.30, 0.10])[0]
    value = "".join(chars)
    for _ in range(operations):
        if not value:
            break
        action = rng.choice(["replace", "delete", "insert", "space", "case"])
        index = rng.randrange(len(value))
        if action == "replace":
            ch = value[index].casefold()
            replacement = rng.choice(CONFUSIONS.get(ch, [rng.choice("abcdefghijklmnopqrstuvwxyz")]))
            value = value[:index] + replacement + value[index + 1:]
        elif action == "delete" and len(value) > 2:
            value = value[:index] + value[index + 1:]
        elif action == "insert":
            value = value[:index] + rng.choice("ilrnt.' ") + value[index:]
        elif action == "space":
            value = value[:index] + (" " if value[index] != " " else "") + value[index:]
        elif action == "case":
            value = value[:index] + value[index].swapcase() + value[index + 1:]
    return value.strip()


def sample_negative(correct: str, corpus: list[tuple[str, float]], rng: random.Random) -> str:
    correct_norm = normalize_direction(correct)
    candidates = [text for text, _ in corpus if normalize_direction(text) != correct_norm]
    # Half hard negatives with similar leading letters/length.
    hard = [item for item in candidates if item[:1].casefold() == correct[:1].casefold() and abs(len(item) - len(correct)) < 8]
    return rng.choice(hard if hard and rng.random() < 0.7 else candidates)


def _group_split(group_count: int, seed: int) -> dict[str, list[int]]:
    order = list(range(group_count))
    random.Random(seed).shuffle(order)
    train_end = int(group_count * 0.70)
    calibration_end = int(group_count * 0.80)
    threshold_end = int(group_count * 0.90)
    return {
        "train": order[:train_end],
        "probability_calibration": order[train_end:calibration_end],
        "safety_audit": order[calibration_end:threshold_end],
        "frozen_test": order[threshold_end:],
    }


def _digit_signature(value: str) -> tuple[str, ...]:
    return tuple(char for char in value if char.isdigit())


def _token_inventory(corpus: list[tuple[str, float]]) -> tuple[dict[str, str], Counter[str]]:
    surfaces: dict[str, str] = {}
    frequencies: Counter[str] = Counter()
    for phrase, phrase_frequency in corpus:
        for token in _WORD_RE.findall(phrase):
            normalized = token.casefold()
            surfaces.setdefault(normalized, token)
            frequencies[normalized] += float(phrase_frequency)
    return surfaces, frequencies


def _sample_token_negative(correct: str, tokens: list[str], rng: random.Random) -> str:
    signature = _digit_signature(correct)
    candidates = [
        token
        for token in tokens
        if token != correct
        and _digit_signature(token) == signature
        and abs(len(token) - len(correct)) <= max(3, int(round(len(correct) * 0.40)))
    ]
    if not candidates:
        candidates = [token for token in tokens if token != correct and _digit_signature(token) == signature]
    if not candidates:
        candidates = [token for token in tokens if token != correct]
    hard = [
        token
        for token in candidates
        if token[:1] == correct[:1] or token[-1:] == correct[-1:]
    ]
    return rng.choice(hard if hard and rng.random() < 0.80 else candidates)


def _pair_examples_for_groups(
    groups: list[int],
    corpus: list[tuple[str, float]],
    frequency_by_text: dict[str, float],
    *,
    seed: int,
    per_phrase: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, float]]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    observed_candidates: list[tuple[str, str, float]] = []
    for group in sorted(groups):
        correct, frequency = corpus[group]
        rng = random.Random(seed + group * 1_000_003)
        for _ in range(per_phrase):
            observed = corrupt(correct, rng, severity=rng.choice((1, 1, 2, 2, 3)))
            rows.append(feature_vector(observed, correct, frequency))
            labels.append(1)
            observed_candidates.append((observed, correct, frequency))
            for _ in range(3):
                negative = sample_negative(correct, corpus, rng)
                negative_frequency = frequency_by_text[negative]
                rows.append(feature_vector(observed, negative, negative_frequency))
                labels.append(0)
                observed_candidates.append((observed, negative, negative_frequency))
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        observed_candidates,
    )


def _pair_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_accepts": int(np.sum((predictions == 1) & (labels == 0))),
        "false_rejects": int(np.sum((predictions == 0) & (labels == 1))),
        "threshold": float(threshold),
    }


def _synthetic_observations(
    phrases: list[str],
    *,
    seed: int,
    per_phrase: int,
    severities: tuple[int, ...] = (1, 1, 2, 2, 3),
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for phrase_index, expected in enumerate(phrases):
        rng = random.Random(seed + phrase_index * 100_003)
        for _ in range(per_phrase):
            rows.append((expected, corrupt(expected, rng, severity=rng.choice(severities))))
    return rows


def _decoder_metrics(
    corrector: DirectionCorrector,
    observations: list[tuple[str, str]],
    *,
    include_examples: bool = False,
    progress_label: str | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    prevented_forced_replacements = 0
    for row_index, (expected, observed) in enumerate(observations, start=1):
        suggestion = corrector.suggest(observed)
        correct = normalize_direction(suggestion.text) == normalize_direction(expected)
        automatic = corrector.should_autocorrect(suggestion)
        legacy_correct = correct
        if suggestion.method == "token-preserve":
            legacy_direct = corrector._direct_suggest(observed)
            legacy_correct = normalize_direction(legacy_direct.text) == normalize_direction(expected)
        if normalize_direction(expected) not in corrector.frequencies:
            compositional = corrector._compositional_suggest(observed)
            if compositional.method == "token-preserve" and normalize_direction(compositional.text) == normalize_direction(expected):
                direct = corrector._direct_suggest(observed)
                if normalize_direction(direct.text) != normalize_direction(expected):
                    prevented_forced_replacements += 1
        rows.append(
            {
                "expected": expected,
                "observed": observed,
                "suggested": suggestion.text,
                "correct": correct,
                "legacy_forced_lexicon_correct": legacy_correct,
                "automatic": automatic,
                "method": suggestion.method,
                "probability": float(suggestion.probability),
                "margin": float(suggestion.margin),
                "edit_ratio": float(suggestion.edit_ratio),
                "safety_class": suggestion.safety_class,
            }
        )
        if progress_label and (row_index % 32 == 0 or row_index == len(observations)):
            print(
                json.dumps({"stage": progress_label, "completed": row_index, "total": len(observations)}),
                flush=True,
            )
    automatic_rows = [row for row in rows if row["automatic"]]
    errors = [row for row in automatic_rows if not row["correct"]]
    sample_count = len(rows)
    result: dict[str, object] = {
        "samples": sample_count,
        "top1_accuracy": sum(bool(row["correct"]) for row in rows) / max(sample_count, 1),
        "legacy_forced_lexicon_top1_accuracy": (
            sum(bool(row["legacy_forced_lexicon_correct"]) for row in rows) / max(sample_count, 1)
        ),
        "autocorrect_count": len(automatic_rows),
        "autocorrect_coverage": len(automatic_rows) / max(sample_count, 1),
        "autocorrect_precision": (
            sum(bool(row["correct"]) for row in automatic_rows) / len(automatic_rows)
            if automatic_rows else 1.0
        ),
        "autocorrect_errors": errors,
        "prevented_forced_lexicon_replacements": prevented_forced_replacements,
        "method_counts": dict(sorted(Counter(str(row["method"]) for row in rows).items())),
    }
    if include_examples:
        result["examples"] = rows[:80]
    return result


def _validate_compositional_phrases(corpus: list[tuple[str, float]], phrases: list[str]) -> None:
    known_phrases = {normalize_direction(text) for text, _frequency in corpus}
    token_surfaces, _frequencies = _token_inventory(corpus)
    errors: list[str] = []
    for phrase in phrases:
        normalized = normalize_direction(phrase)
        if normalized in known_phrases:
            errors.append(f"phrase unexpectedly belongs to lexicon: {phrase}")
        unknown = [token for token in _WORD_RE.findall(phrase) if token.casefold() not in token_surfaces]
        if unknown:
            errors.append(f"unknown tokens in {phrase!r}: {unknown}")
    if errors:
        raise ValueError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "src/scorescan/resources/direction_model.json")
    parser.add_argument("--report", type=Path, default=ROOT.parent / "training/direction_model_report_v9.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT.parent / "training/baselines/direction_model_v8.json")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--ocr-observations", type=Path, default=ROOT.parent / "training/direction_rendered_ocr_v2.jsonl")
    parser.add_argument(
        "--compositional-ocr-observations",
        type=Path,
        default=ROOT.parent / "training/direction_compositional_rendered_ocr_v1.jsonl",
    )
    parser.add_argument("--samples-per-term", type=int, default=3)
    parser.add_argument("--token-samples", type=int, default=4)
    parser.add_argument("--eval-per-term", type=int, default=3)
    args = parser.parse_args()

    corpus = build_corpus()
    phrase_index = {normalize_direction(text): index for index, (text, _frequency) in enumerate(corpus)}
    frequency_by_text = dict(corpus)
    splits = _group_split(len(corpus), args.seed)
    train_groups = set(splits["train"])
    _validate_compositional_phrases(corpus, COMPOSITIONAL_THRESHOLD + COMPOSITIONAL_TEST)

    def load_observations(path: Path | None) -> list[tuple[str, str]]:
        observations: list[tuple[str, str]] = []
        if not path or not path.exists():
            return observations
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record {path}:{line_number}") from exc
            expected = str(item.get("expected", "")).strip()
            observed = str(item.get("observed", "")).strip()
            if expected and observed:
                observations.append((expected, observed))
        return observations

    rendered_observations = [
        (expected, observed)
        for expected, observed in load_observations(args.ocr_observations)
        if normalize_direction(expected) in phrase_index
    ]
    compositional_rendered_observations = load_observations(args.compositional_ocr_observations)
    expected_compositional = {normalize_direction(value) for value in COMPOSITIONAL_TEST}
    unexpected_compositional = sorted({
        expected
        for expected, _observed in compositional_rendered_observations
        if normalize_direction(expected) not in expected_compositional
    })
    if unexpected_compositional:
        raise ValueError(
            "compositional OCR dataset contains phrases outside COMPOSITIONAL_TEST: "
            + ", ".join(unexpected_compositional[:5])
        )

    rows: list[list[float]] = []
    labels: list[int] = []
    training_kind_counts: Counter[str] = Counter()
    for group in sorted(train_groups):
        correct, frequency = corpus[group]
        rng = random.Random(args.seed + group * 1_000_003)
        observations = [correct]
        observations.extend(
            corrupt(correct, rng, severity=rng.choice((1, 1, 2, 2, 3)))
            for _ in range(args.samples_per_term)
        )
        for observed in observations:
            rows.append(feature_vector(observed, correct, frequency))
            labels.append(1)
            training_kind_counts["phrase_positive"] += 1
            for _ in range(3):
                negative = sample_negative(correct, corpus, rng)
                rows.append(feature_vector(observed, negative, frequency_by_text[negative]))
                labels.append(0)
                training_kind_counts["phrase_negative"] += 1

    # Real rendered observations are training evidence only when their expected phrase
    # belongs to the fitting partition.  The same phrase can never enter a held-out split.
    for record_index, (expected, observed) in enumerate(rendered_observations):
        group = phrase_index[normalize_direction(expected)]
        if group not in train_groups:
            continue
        rng = random.Random(args.seed + 70_000_019 + record_index * 1009)
        rows.append(feature_vector(observed, expected, frequency_by_text[expected]))
        labels.append(1)
        training_kind_counts["rendered_positive"] += 1
        for _ in range(3):
            negative = sample_negative(expected, corpus, rng)
            rows.append(feature_vector(observed, negative, frequency_by_text[negative]))
            labels.append(0)
            training_kind_counts["rendered_negative"] += 1

    # The deployed pair model also ranks individual words in novel phrases.  Earlier
    # releases trained it only on complete phrases, which was a domain mismatch.  Token
    # examples use the closed runtime vocabulary and independent corruptions.
    token_surfaces, token_frequencies = _token_inventory(corpus)
    token_norms = sorted(token_surfaces)
    for token_index, correct_norm in enumerate(token_norms):
        correct = token_surfaces[correct_norm]
        frequency = float(token_frequencies[correct_norm])
        rng = random.Random(args.seed + 90_000_007 + token_index * 1_000_003)
        for _ in range(args.token_samples):
            observed = corrupt(correct, rng, severity=rng.choice((1, 1, 1, 2)))
            rows.append(feature_vector(observed, correct, frequency))
            labels.append(1)
            training_kind_counts["token_positive"] += 1
            for _ in range(3):
                negative_norm = _sample_token_negative(correct_norm, token_norms, rng)
                negative = token_surfaces[negative_norm]
                rows.append(feature_vector(observed, negative, float(token_frequencies[negative_norm])))
                labels.append(0)
                training_kind_counts["token_negative"] += 1

    x_train = np.asarray(rows, dtype=np.float64)
    y_train = np.asarray(labels, dtype=np.int64)
    means = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales < 1e-9] = 1.0
    x_standardized = (x_train - means) / scales
    model = LogisticRegression(
        max_iter=600,
        class_weight="balanced",
        random_state=args.seed,
        solver="liblinear",
    )
    model.fit(x_standardized, y_train)

    # Fit an independent one-dimensional Platt calibrator on phrase identities that
    # never participated in the base model.  The affine calibration is folded into
    # the deployed coefficients, so the runtime remains a single dependency-free
    # logistic evaluation.
    x_calibration, y_calibration, _calibration_pairs = _pair_examples_for_groups(
        splits["probability_calibration"],
        corpus,
        frequency_by_text,
        seed=args.seed + 100_000_019,
        per_phrase=args.eval_per_term,
    )
    calibration_scores = model.decision_function((x_calibration - means) / scales).reshape(-1, 1)
    calibrator = LogisticRegression(
        max_iter=300,
        random_state=args.seed,
        solver="liblinear",
    )
    calibrator.fit(calibration_scores, y_calibration)
    calibration_slope = float(calibrator.coef_[0][0])
    calibration_intercept = float(calibrator.intercept_[0])
    deployed_intercept = calibration_slope * float(model.intercept_[0]) + calibration_intercept
    deployed_coefficients = calibration_slope * model.coef_[0]

    payload = {
        "format": 3,
        "model_version": "scorescan-direction-logistic-9",
        "seed": args.seed,
        "feature_names": list(DIRECTION_FEATURE_NAMES),
        "intercept": deployed_intercept,
        "coefficients": [float(value) for value in deployed_coefficients],
        "means": [float(value) for value in means],
        "scales": [float(value) for value in scales],
        "probability_calibration": {
            "method": "platt-logistic",
            "group_count": len(splits["probability_calibration"]),
            "samples": int(y_calibration.size),
            "slope": calibration_slope,
            "intercept": calibration_intercept,
        },
        "lexicon": [{"text": text, "frequency": frequency} for text, frequency in corpus],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "model_trained", "training_samples": int(y_train.size)}), flush=True)

    corrector = DirectionCorrector(args.output, verify_manifest=False)
    baseline = DirectionCorrector(args.baseline_model, verify_manifest=False)
    if not corrector.model_enabled:
        raise RuntimeError(f"serialized direction model did not load: {corrector.model_status}")

    # Pair-level frozen test uses unseen phrase identities and independent corruptions.
    x_test, y_test, pair_observed_candidates = _pair_examples_for_groups(
        splits["frozen_test"],
        corpus,
        frequency_by_text,
        seed=args.seed + 110_000_021,
        per_phrase=args.eval_per_term,
    )
    base_test_scores = model.decision_function((x_test - means) / scales)
    calibrated_test_scores = calibration_slope * base_test_scores + calibration_intercept
    candidate_probabilities = 1.0 / (1.0 + np.exp(-np.clip(calibrated_test_scores, -40.0, 40.0)))
    baseline_probabilities = np.asarray(
        [baseline._probability(observed, candidate, frequency) for observed, candidate, frequency in pair_observed_candidates],
        dtype=np.float64,
    )
    deployed_probabilities = np.asarray(
        [corrector._probability(observed, candidate, frequency) for observed, candidate, frequency in pair_observed_candidates],
        dtype=np.float64,
    )
    print(json.dumps({"stage": "pair_evaluation", "samples": int(y_test.size)}), flush=True)

    phrase_safety_observations = _synthetic_observations(
        [corpus[group][0] for group in sorted(splits["safety_audit"])],
        seed=args.seed + 115_000_003,
        per_phrase=args.eval_per_term,
    )
    phrase_test_observations = _synthetic_observations(
        [corpus[group][0] for group in sorted(splits["frozen_test"])],
        seed=args.seed + 120_000_007,
        per_phrase=args.eval_per_term,
    )
    compositional_threshold_observations = _synthetic_observations(
        COMPOSITIONAL_THRESHOLD,
        seed=args.seed + 130_000_009,
        per_phrase=8,
        severities=(1, 1, 1, 2),
    )
    compositional_test_observations = _synthetic_observations(
        COMPOSITIONAL_TEST,
        seed=args.seed + 140_000_011,
        per_phrase=8,
        severities=(1, 1, 1, 2),
    )
    frozen_group_set = set(splits["frozen_test"])
    rendered_frozen = [
        (expected, observed)
        for expected, observed in rendered_observations
        if phrase_index[normalize_direction(expected)] in frozen_group_set
    ]

    print(json.dumps({"stage": "decoder_evaluation", "rendered_frozen": len(rendered_frozen)}), flush=True)
    phrase_safety_audit = _decoder_metrics(corrector, phrase_safety_observations, progress_label="phrase_safety_audit")
    print(json.dumps({"stage": "phrase_safety_audit", "top1": phrase_safety_audit["top1_accuracy"]}), flush=True)
    phrase_candidate = _decoder_metrics(corrector, phrase_test_observations, progress_label="phrase_candidate")
    print(json.dumps({"stage": "phrase_candidate", "top1": phrase_candidate["top1_accuracy"]}), flush=True)
    phrase_baseline = _decoder_metrics(baseline, phrase_test_observations, progress_label="phrase_baseline")
    print(json.dumps({"stage": "phrase_baseline", "top1": phrase_baseline["top1_accuracy"]}), flush=True)
    compositional_threshold_candidate = _decoder_metrics(corrector, compositional_threshold_observations, progress_label="compositional_threshold_candidate")
    print(json.dumps({"stage": "compositional_threshold_candidate", "top1": compositional_threshold_candidate["top1_accuracy"]}), flush=True)
    compositional_threshold_baseline = _decoder_metrics(baseline, compositional_threshold_observations, progress_label="compositional_threshold_baseline")
    print(json.dumps({"stage": "compositional_threshold_baseline", "top1": compositional_threshold_baseline["top1_accuracy"]}), flush=True)
    compositional_test_candidate = _decoder_metrics(corrector, compositional_test_observations, include_examples=True, progress_label="compositional_test_candidate")
    print(json.dumps({"stage": "compositional_test_candidate", "top1": compositional_test_candidate["top1_accuracy"]}), flush=True)
    compositional_test_baseline = _decoder_metrics(baseline, compositional_test_observations, progress_label="compositional_test_baseline")
    print(json.dumps({"stage": "compositional_test_baseline", "top1": compositional_test_baseline["top1_accuracy"]}), flush=True)
    rendered_candidate = _decoder_metrics(corrector, rendered_frozen, include_examples=True, progress_label="rendered_candidate")
    print(json.dumps({"stage": "rendered_candidate", "top1": rendered_candidate["top1_accuracy"]}), flush=True)
    rendered_baseline = _decoder_metrics(baseline, rendered_frozen, progress_label="rendered_baseline")
    print(json.dumps({"stage": "rendered_baseline", "top1": rendered_baseline["top1_accuracy"]}), flush=True)
    compositional_rendered_candidate = _decoder_metrics(
        corrector,
        compositional_rendered_observations,
        include_examples=True,
        progress_label="compositional_rendered_candidate",
    )
    print(
        json.dumps(
            {"stage": "compositional_rendered_candidate", "top1": compositional_rendered_candidate["top1_accuracy"]}
        ),
        flush=True,
    )
    compositional_rendered_baseline = _decoder_metrics(
        baseline,
        compositional_rendered_observations,
        progress_label="compositional_rendered_baseline",
    )
    print(
        json.dumps(
            {"stage": "compositional_rendered_baseline", "top1": compositional_rendered_baseline["top1_accuracy"]}
        ),
        flush=True,
    )

    report = {
        "model_version": payload["model_version"],
        "seed": args.seed,
        "lexicon_size": len(corpus),
        "token_vocabulary": len(token_norms),
        "training_samples": int(y_train.size),
        "training_kind_counts": dict(sorted(training_kind_counts.items())),
        "splits": {key: len(value) for key, value in splits.items()},
        "probability_calibration": {
            "method": "platt-logistic",
            "samples": int(y_calibration.size),
            "slope": calibration_slope,
            "intercept": calibration_intercept,
        },
        "phrase_safety_audit": phrase_safety_audit,
        "pair_frozen_test": _pair_metrics(y_test, candidate_probabilities),
        "baseline_v8_pair_same_frozen_test": _pair_metrics(y_test, baseline_probabilities),
        "deployment_parity": {
            "samples": int(y_test.size),
            "max_absolute_probability_delta": float(np.max(np.abs(candidate_probabilities - deployed_probabilities))),
        },
        "phrase_decoder_frozen_test": phrase_candidate,
        "baseline_v8_phrase_decoder_same_test": phrase_baseline,
        "compositional_threshold_audit": compositional_threshold_candidate,
        "baseline_v8_compositional_threshold_same_decoder": compositional_threshold_baseline,
        "compositional_frozen_test": compositional_test_candidate,
        "baseline_v8_compositional_same_decoder": compositional_test_baseline,
        "rendered_ocr_frozen_test": rendered_candidate,
        "baseline_v8_rendered_ocr_same_test": rendered_baseline,
        "compositional_rendered_ocr_frozen_test": compositional_rendered_candidate,
        "baseline_v8_compositional_rendered_ocr_same_decoder": compositional_rendered_baseline,
        "compositional_rendered_ocr_dataset": {
            "file": args.compositional_ocr_observations.name if args.compositional_ocr_observations else None,
            "records": len(compositional_rendered_observations),
            "raw_exact": sum(
                normalize_direction(expected) == normalize_direction(observed)
                for expected, observed in compositional_rendered_observations
            ) / max(len(compositional_rendered_observations), 1),
        },
        "rendered_ocr_dataset": {
            "file": args.ocr_observations.name if args.ocr_observations else None,
            "records": len(rendered_observations),
            "frozen_records": len(rendered_frozen),
            "frozen_raw_exact": sum(
                normalize_direction(expected) == normalize_direction(observed)
                for expected, observed in rendered_frozen
            ) / max(len(rendered_frozen), 1),
        },
        "scope": (
            "Grouped phrase matching, deployed direction correction and rendered text OCR. "
            "These are not end-to-end score-recognition metrics."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"compositional_frozen_test", "rendered_ocr_frozen_test"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
