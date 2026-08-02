from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .model_registry import load_verified_json

_WORD_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)
_TOKEN_WITH_AFFIX_RE = re.compile(r"^([^\wÀ-ÖØ-öø-ÿ]*)([A-Za-zÀ-ÖØ-öø-ÿ]+)([^\wÀ-ÖØ-öø-ÿ]*)$", re.UNICODE)

DIRECTION_FEATURE_NAMES = (
    "edit_similarity",
    "trigram_jaccard",
    "token_overlap",
    "prefix",
    "suffix",
    "length_ratio",
    "exact",
    "prefix_ratio",
    "suffix_ratio",
    "char_overlap",
    "space_similarity",
    "damerau_similarity",
    "compact_similarity",
    "skeleton_similarity",
    "accent_fold_similarity",
    "token_count_similarity",
    "punctuation_similarity",
    "token_sequence_similarity",
    "initials_similarity",
    "bigram_similarity",
    "fourgram_similarity",
    "digit_pattern_similarity",
    "log_frequency",
)

_CONNECTOR_TOKENS = frozenset({
    "a", "e", "et", "and", "und", "ma", "al", "da", "de", "in", "au", "en", "un", "con",
})
_SINGLE_TOKEN_DYNAMICS = frozenset({"p", "f"})
_ABBREVIATION_MAP = {
    "rit": "rit.",
    "rall": "rall.",
    "accel": "accel.",
    "cresc": "cresc.",
    "dim": "dim.",
    "decresc": "decresc.",
    "pizz": "pizz.",
}


def normalize_direction(text: str) -> str:
    value = text.strip().replace("’", "'").replace("`", "'")
    value = value.replace("—", "-").replace("–", "-")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^[.,;:!?]+", "", value)
    return value.strip(" \t\r\n,;:()[]{}").casefold()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]



def _damerau_levenshtein(a: str, b: str) -> int:
    """Optimal-string-alignment distance for common adjacent OCR transpositions."""
    rows, cols = len(a) + 1, len(b) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost)
    return matrix[-1][-1]


def _accent_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _ocr_skeleton(value: str) -> str:
    value = _accent_fold(value.casefold())
    replacements = (("rn", "m"), ("cl", "d"), ("vv", "w"))
    for source, target in replacements:
        value = value.replace(source, target)
    table = str.maketrans({"0": "o", "1": "l", "|": "l", "5": "s", "2": "z"})
    value = value.translate(table)
    return "".join(char for char in value if char.isalnum())

def _ngrams(value: str, n: int = 3) -> set[str]:
    padded = f"  {value}  "
    return {padded[i:i+n] for i in range(max(1, len(padded) - n + 1))}


def _token_set(value: str) -> set[str]:
    return set(_WORD_RE.findall(value))


def _sequence_edit_similarity(left: list[str], right: list[str]) -> float:
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return 1.0 if not left else 0.0
    previous = list(range(len(right) + 1))
    for i, token_left in enumerate(left, start=1):
        current = [i]
        for j, token_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token_left != token_right)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def feature_vector(observed: str, candidate: str, frequency: float = 1.0) -> list[float]:
    observed = normalize_direction(observed)
    candidate = normalize_direction(candidate)
    maximum = max(len(observed), len(candidate), 1)
    edit_similarity = 1.0 - _levenshtein(observed, candidate) / maximum
    obs_grams, cand_grams = _ngrams(observed), _ngrams(candidate)
    union = obs_grams | cand_grams
    trigram = len(obs_grams & cand_grams) / max(len(union), 1)
    obs_tokens, cand_tokens = _token_set(observed), _token_set(candidate)
    token_union = obs_tokens | cand_tokens
    token_overlap = len(obs_tokens & cand_tokens) / max(len(token_union), 1)
    prefix = float(observed[:3] == candidate[:3]) if len(observed) >= 3 and len(candidate) >= 3 else 0.0
    suffix = float(observed[-3:] == candidate[-3:]) if len(observed) >= 3 and len(candidate) >= 3 else 0.0
    length_ratio = min(len(observed), len(candidate)) / maximum
    exact = float(observed == candidate)
    prefix_length = 0
    for left, right in zip(observed, candidate):
        if left != right:
            break
        prefix_length += 1
    suffix_length = 0
    for left, right in zip(reversed(observed), reversed(candidate)):
        if left != right:
            break
        suffix_length += 1
    prefix_ratio = prefix_length / maximum
    suffix_ratio = suffix_length / maximum
    obs_chars, cand_chars = set(observed), set(candidate)
    char_union = obs_chars | cand_chars
    char_overlap = len(obs_chars & cand_chars) / max(len(char_union), 1)
    space_similarity = 1.0 - min(1.0, abs(observed.count(" ") - candidate.count(" ")) / max(candidate.count(" ") + 1, 1))
    damerau_similarity = 1.0 - _damerau_levenshtein(observed, candidate) / maximum
    observed_compact = "".join(char for char in observed if char.isalnum())
    candidate_compact = "".join(char for char in candidate if char.isalnum())
    compact_maximum = max(len(observed_compact), len(candidate_compact), 1)
    compact_similarity = 1.0 - _levenshtein(observed_compact, candidate_compact) / compact_maximum
    observed_skeleton, candidate_skeleton = _ocr_skeleton(observed), _ocr_skeleton(candidate)
    skeleton_maximum = max(len(observed_skeleton), len(candidate_skeleton), 1)
    skeleton_similarity = 1.0 - _levenshtein(observed_skeleton, candidate_skeleton) / skeleton_maximum
    observed_folded, candidate_folded = _accent_fold(observed), _accent_fold(candidate)
    folded_maximum = max(len(observed_folded), len(candidate_folded), 1)
    accent_fold_similarity = 1.0 - _levenshtein(observed_folded, candidate_folded) / folded_maximum
    observed_token_count = len(_WORD_RE.findall(observed))
    candidate_token_count = len(_WORD_RE.findall(candidate))
    token_count_similarity = 1.0 - min(1.0, abs(observed_token_count - candidate_token_count) / max(candidate_token_count, 1))
    punctuation = set(".,;:'=-")
    observed_punctuation = {char for char in observed if char in punctuation}
    candidate_punctuation = {char for char in candidate if char in punctuation}
    punctuation_union = observed_punctuation | candidate_punctuation
    punctuation_similarity = (
        len(observed_punctuation & candidate_punctuation) / len(punctuation_union)
        if punctuation_union else 1.0
    )
    observed_tokens_ordered = [token.casefold() for token in _WORD_RE.findall(observed)]
    candidate_tokens_ordered = [token.casefold() for token in _WORD_RE.findall(candidate)]
    token_sequence_similarity = _sequence_edit_similarity(observed_tokens_ordered, candidate_tokens_ordered)
    observed_initials = "".join(token[:1] for token in observed_tokens_ordered)
    candidate_initials = "".join(token[:1] for token in candidate_tokens_ordered)
    initials_maximum = max(len(observed_initials), len(candidate_initials), 1)
    initials_similarity = 1.0 - _levenshtein(observed_initials, candidate_initials) / initials_maximum
    obs_bigrams, cand_bigrams = _ngrams(observed, 2), _ngrams(candidate, 2)
    bigram_similarity = len(obs_bigrams & cand_bigrams) / max(len(obs_bigrams | cand_bigrams), 1)
    obs_fourgrams, cand_fourgrams = _ngrams(observed, 4), _ngrams(candidate, 4)
    fourgram_similarity = len(obs_fourgrams & cand_fourgrams) / max(len(obs_fourgrams | cand_fourgrams), 1)
    digit_pattern_similarity = float(
        [char for char in observed if char.isdigit()] == [char for char in candidate if char.isdigit()]
    )
    return [
        edit_similarity,
        trigram,
        token_overlap,
        prefix,
        suffix,
        length_ratio,
        exact,
        prefix_ratio,
        suffix_ratio,
        char_overlap,
        space_similarity,
        damerau_similarity,
        compact_similarity,
        skeleton_similarity,
        accent_fold_similarity,
        token_count_similarity,
        punctuation_similarity,
        token_sequence_similarity,
        initials_similarity,
        bigram_similarity,
        fourgram_similarity,
        digit_pattern_similarity,
        math.log1p(max(0.0, frequency)),
    ]


@dataclass(frozen=True)
class DirectionSuggestion:
    text: str
    probability: float
    margin: float
    changed: bool
    method: str = "lexicon"
    autocorrect_safe: bool = False
    edit_ratio: float = 1.0
    safety_class: str = "none"


class DirectionCorrector:
    """Conservative musical-direction corrector.

    Version 9 combines whole-phrase ranking with a token-level decoder.  The latter
    supports previously unseen combinations such as ``Allegro molto cantabile`` while
    preserving unfamiliar words instead of forcing the entire phrase into a finite
    dictionary entry.
    """

    def __init__(self, model_path: Path | None = None, *, verify_manifest: bool = True) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "resources" / "direction_model.json"
        self.model_path = model_path
        if verify_manifest:
            loaded = load_verified_json(model_path, "music_direction_correction")
            payload = loaded.payload
            self.model_verified = loaded.verified
            self.model_status = loaded.status
        else:
            try:
                raw_payload = json.loads(model_path.read_text(encoding="utf-8"))
                payload = raw_payload if isinstance(raw_payload, dict) else {}
            except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            self.model_verified = False
            self.model_status = "verification_bypassed"

        self.model_format = int(payload.get("format", 0) or 0) if isinstance(payload, dict) else 0
        self.model_version = str(payload.get("model_version", "disabled")) if isinstance(payload, dict) else "disabled"
        declared_features = tuple(str(value) for value in payload.get("feature_names", ())) if isinstance(payload, dict) else ()
        try:
            intercept = float(payload.get("intercept", 0.0))
            coefficients = tuple(float(value) for value in payload.get("coefficients", ()))
            if self.model_format >= 3:
                means = tuple(float(value) for value in payload.get("means", ()))
                scales = tuple(float(value) for value in payload.get("scales", ()))
            else:
                means = (0.0,) * len(DIRECTION_FEATURE_NAMES)
                scales = (1.0,) * len(DIRECTION_FEATURE_NAMES)
        except (TypeError, ValueError, OverflowError):
            intercept = 0.0
            coefficients = ()
            means = ()
            scales = ()

        numeric_values = (intercept,) + coefficients + means + scales
        model_shape_valid = (
            self.model_format in {2, 3}
            and declared_features == DIRECTION_FEATURE_NAMES
            and len(coefficients) == len(DIRECTION_FEATURE_NAMES)
            and len(means) == len(DIRECTION_FEATURE_NAMES)
            and len(scales) == len(DIRECTION_FEATURE_NAMES)
            and all(math.isfinite(value) for value in numeric_values)
            and all(value > 0.0 for value in scales)
        )

        entries = payload.get("lexicon", ()) if isinstance(payload, dict) else ()
        lexicon: list[str] = []
        frequencies: dict[str, float] = {}
        if isinstance(entries, list):
            for entry in entries:
                try:
                    if isinstance(entry, str):
                        text, frequency = entry, 1.0
                    elif isinstance(entry, dict):
                        text = str(entry.get("text", ""))
                        frequency = float(entry.get("frequency", 1.0))
                    else:
                        continue
                except (TypeError, ValueError, OverflowError):
                    continue
                normalized = normalize_direction(text)
                if not normalized or not math.isfinite(frequency) or frequency < 0.0:
                    continue
                if normalized not in frequencies:
                    lexicon.append(text)
                    frequencies[normalized] = frequency

        self.model_enabled = bool(model_shape_valid and lexicon)
        if not self.model_enabled:
            self.model_status = f"{self.model_status}:invalid_direction_model"
            intercept = 0.0
            coefficients = (0.0,) * len(DIRECTION_FEATURE_NAMES)
            means = (0.0,) * len(DIRECTION_FEATURE_NAMES)
            scales = (1.0,) * len(DIRECTION_FEATURE_NAMES)
            lexicon = []
            frequencies = {}

        self.intercept = intercept
        self.coefficients = coefficients
        self.means = means
        self.scales = scales
        self.lexicon = lexicon
        self.frequencies = frequencies
        self._normalized = [normalize_direction(item) for item in self.lexicon]
        self._grams = [_ngrams(item) for item in self._normalized]
        self._phrase_gram_index: dict[str, set[int]] = {}
        self._phrase_length_index: dict[int, list[int]] = {}
        self._phrase_initial_index: dict[str, set[int]] = {}
        for index, (normalized, grams) in enumerate(zip(self._normalized, self._grams, strict=True)):
            for gram in grams:
                self._phrase_gram_index.setdefault(gram, set()).add(index)
            self._phrase_length_index.setdefault(len(normalized), []).append(index)
            if normalized:
                self._phrase_initial_index.setdefault(normalized[0], set()).add(index)

        token_surfaces: dict[str, tuple[str, float]] = {}
        token_allowed_affixes: dict[str, set[tuple[str, str]]] = {}
        for phrase in self.lexicon:
            frequency = self.frequencies.get(normalize_direction(phrase), 1.0)
            for raw_chunk in phrase.split():
                match = _TOKEN_WITH_AFFIX_RE.match(raw_chunk)
                if not match:
                    continue
                prefix, token, suffix = match.groups()
                normalized = token.casefold()
                existing = token_surfaces.get(normalized)
                if existing is None or frequency > existing[1]:
                    token_surfaces[normalized] = (token, frequency)
                token_allowed_affixes.setdefault(normalized, set()).add((prefix, suffix))
        self._token_surfaces = token_surfaces
        self._token_allowed_affixes = token_allowed_affixes
        self._token_norms = sorted(token_surfaces)
        self._token_grams = {token: _ngrams(token) for token in self._token_norms}
        self._token_gram_index: dict[str, set[str]] = {}
        self._token_length_index: dict[int, list[str]] = {}
        self._token_initial_index: dict[str, set[str]] = {}
        for token, grams in self._token_grams.items():
            for gram in grams:
                self._token_gram_index.setdefault(gram, set()).add(token)
            self._token_length_index.setdefault(len(token), []).append(token)
            if token:
                self._token_initial_index.setdefault(token[0], set()).add(token)

    def _probability(self, observed: str, candidate: str, frequency: float | None = None) -> float:
        if not self.model_enabled:
            return 0.0
        frequency = frequency if frequency is not None else self.frequencies.get(normalize_direction(candidate), 1.0)
        raw_features = feature_vector(observed, candidate, frequency)
        features = tuple(
            (float(value) - mean) / scale
            for value, mean, scale in zip(raw_features, self.means, self.scales, strict=True)
        )
        if not all(math.isfinite(value) for value in features):
            return 0.0
        score = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, features, strict=True)
        )
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-min(score, 40)))
        exp = math.exp(max(score, -40))
        return exp / (1.0 + exp)

    @staticmethod
    def _digit_signature(value: str) -> tuple[str, ...]:
        return tuple(char for char in value if char.isdigit())

    @classmethod
    def _digit_pattern_compatible(cls, observed: str, candidate: str) -> bool:
        # Numeric tempo and navigation text is high-impact.  A language model must not
        # silently turn alphabetic OCR into a number (``vnd`` -> ``2nd``) or invent a
        # different metronome value.  Such cases remain review suggestions.
        return cls._digit_signature(observed) == cls._digit_signature(candidate)

    def _affix_allowed(self, token: str, prefix: str, suffix: str) -> bool:
        if not prefix and not suffix:
            return True
        return (prefix, suffix) in self._token_allowed_affixes.get(token.casefold(), set())

    def _candidate_phrase_indices(self, normalized: str, grams: set[str], limit: int) -> set[int]:
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(self._phrase_gram_index.get(gram, ()))
        if normalized:
            candidates.update(self._phrase_initial_index.get(normalized[0], ()))
        length_window = max(4, int(round(len(normalized) * 0.35)))
        for length in range(max(1, len(normalized) - length_window), len(normalized) + length_window + 1):
            candidates.update(self._phrase_length_index.get(length, ()))
        # Severe OCR damage can eliminate every useful trigram.  Falling back to the
        # complete lexicon preserves recall while the normal path remains sub-linear.
        if len(candidates) < limit:
            candidates.update(range(len(self.lexicon)))
        return candidates

    def _shortlist(self, observed: str, limit: int = 48) -> list[str]:
        normalized = normalize_direction(observed)
        if not normalized or not self.lexicon:
            return []
        grams = _ngrams(normalized)
        scores: list[tuple[float, str]] = []
        for index in self._candidate_phrase_indices(normalized, grams, limit):
            candidate = self.lexicon[index]
            cand_norm = self._normalized[index]
            cand_grams = self._grams[index]
            union = grams | cand_grams
            jaccard = len(grams & cand_grams) / max(len(union), 1)
            length = min(len(normalized), len(cand_norm)) / max(len(normalized), len(cand_norm), 1)
            first = 0.15 if normalized[:1] == cand_norm[:1] else 0.0
            scores.append((jaccard * 0.7 + length * 0.3 + first, candidate))
        scores.sort(reverse=True, key=lambda item: item[0])
        return [candidate for _, candidate in scores[:limit]]

    def _candidate_tokens(self, normalized: str, grams: set[str], limit: int) -> set[str]:
        candidates: set[str] = set()
        for gram in grams:
            candidates.update(self._token_gram_index.get(gram, ()))
        if normalized:
            candidates.update(self._token_initial_index.get(normalized[0], ()))
        length_window = max(2, int(round(len(normalized) * 0.35)))
        for length in range(max(1, len(normalized) - length_window), len(normalized) + length_window + 1):
            candidates.update(self._token_length_index.get(length, ()))
        if len(candidates) < limit:
            candidates.update(self._token_norms)
        return candidates

    def _token_shortlist(self, observed: str, limit: int = 20) -> list[str]:
        normalized = observed.casefold()
        grams = _ngrams(normalized)
        scored: list[tuple[float, str]] = []
        for token in self._candidate_tokens(normalized, grams, limit):
            candidate_grams = self._token_grams[token]
            jaccard = len(grams & candidate_grams) / max(len(grams | candidate_grams), 1)
            length = min(len(normalized), len(token)) / max(len(normalized), len(token), 1)
            first = 0.12 if normalized[:1] == token[:1] else 0.0
            scored.append((0.72 * jaccard + 0.28 * length + first, token))
        scored.sort(reverse=True)
        return [token for _, token in scored[:limit]]

    @lru_cache(maxsize=4096)
    def _direct_suggest(self, observed: str) -> DirectionSuggestion:
        normalized = normalize_direction(observed)
        if not normalized:
            return DirectionSuggestion(observed, 0.0, 0.0, False, "empty", False, 0.0)
        candidates = self._shortlist(observed)
        if not candidates:
            return DirectionSuggestion(observed, 0.0, 0.0, False, "none", False, 1.0)
        observed_token_count = len(_WORD_RE.findall(observed))
        ranked_items: list[tuple[float, str]] = []
        for item in candidates:
            probability = self._probability(observed, item)
            if not self._digit_pattern_compatible(observed, item):
                probability *= 0.05
            candidate_token_count = len(_WORD_RE.findall(item))
            token_gap = abs(observed_token_count - candidate_token_count)
            if observed_token_count >= 2 and token_gap:
                probability *= max(0.30, 1.0 - 0.22 * token_gap)
            ranked_items.append((probability, item))
        ranked = sorted(ranked_items, reverse=True)
        best_probability, best_text = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        margin = best_probability - second
        changed = normalize_direction(best_text) != normalized
        edit_ratio = _levenshtein(normalized, normalize_direction(best_text)) / max(
            len(normalized), len(normalize_direction(best_text)), 1
        )
        observed_tokens = len(_WORD_RE.findall(observed))
        candidate_tokens = len(_WORD_RE.findall(best_text))
        safe = (
            changed
            and observed_tokens == candidate_tokens
            and normalized[:1] == normalize_direction(best_text)[:1]
            and edit_ratio <= 0.24
            and best_probability >= 0.985
            and margin >= 0.24
        )
        return DirectionSuggestion(
            best_text,
            best_probability,
            margin,
            changed,
            "phrase",
            safe,
            edit_ratio,
            "known-phrase" if safe else "none",
        )

    def _merge_split_tokens(self, chunks: list[str]) -> tuple[list[str], bool]:
        """Merge OCR-split words without crossing established token boundaries.

        Character-preserving exact joins are considered safe only when at least one
        fragment is unknown and neither fragment is a one-character token.  Fuzzy joins
        are retained as review suggestions but are never eligible for automatic write-back.
        """
        merged: list[str] = []
        structural_safe = True
        index = 0
        while index < len(chunks):
            if index + 1 < len(chunks):
                first = _TOKEN_WITH_AFFIX_RE.match(chunks[index])
                second = _TOKEN_WITH_AFFIX_RE.match(chunks[index + 1])
                if first and second:
                    first_prefix, first_core, first_suffix = first.groups()
                    second_prefix, second_core, second_suffix = second.groups()
                    first_known = first_core.casefold() in self._token_surfaces
                    second_known = second_core.casefold() in self._token_surfaces
                    punctuation_ok = first_suffix in {"", ".", "'"} and second_prefix in {"", ".", "'"}
                    joined = first_core + second_core
                    joined_norm = joined.casefold()
                    exact_join = joined_norm in self._token_surfaces
                    # Never join two already valid words (e.g. ``con a`` -> ``coda``).
                    can_join = punctuation_ok and not (first_known and second_known)
                    if can_join and (exact_join or (len(first_core) >= 2 and len(second_core) >= 2)):
                        ranked: list[tuple[float, str]] = []
                        if exact_join:
                            ranked.append((1.0, joined_norm))
                        else:
                            for candidate_norm in self._token_shortlist(joined, limit=10):
                                surface, frequency = self._token_surfaces[candidate_norm]
                                ranked.append((self._probability(joined, surface, frequency), candidate_norm))
                            ranked.sort(reverse=True)
                        if ranked:
                            probability, candidate_norm = ranked[0]
                            second_probability = ranked[1][0] if len(ranked) > 1 else 0.0
                            edit_distance = _levenshtein(joined_norm, candidate_norm)
                            if probability >= 0.96 and probability - second_probability >= 0.12 and edit_distance <= 1:
                                surface = self._token_surfaces[candidate_norm][0]
                                surface = surface[:1].upper() + surface[1:] if first_core[:1].isupper() else surface.casefold()
                                merged.append(first_prefix + surface + second_suffix)
                                # One-character fragments are too ambiguous for unattended
                                # correction (they may be a missing dynamic or connector).
                                structural_safe = structural_safe and exact_join and min(len(first_core), len(second_core)) >= 2
                                index += 2
                                continue
            merged.append(chunks[index])
            index += 1
        return merged, structural_safe

    @lru_cache(maxsize=4096)
    def _compositional_suggest(self, observed: str) -> DirectionSuggestion:
        original_chunks = observed.strip().split()
        cleaned_chunks: list[str] = []
        precleaned = False
        for chunk in original_chunks:
            internal_noise = re.fullmatch(r"([A-Za-zÀ-ÖØ-öø-ÿ]{1,})[.']([A-Za-zÀ-ÖØ-öø-ÿ]{1,})", chunk)
            if internal_noise:
                cleaned_chunks.append(internal_noise.group(1) + internal_noise.group(2))
                precleaned = True
            else:
                cleaned_chunks.append(chunk)
        chunks, merge_safe = self._merge_split_tokens(cleaned_chunks)
        premerged = normalize_direction(" ".join(chunks)) != normalize_direction(observed)
        structural_safe = merge_safe and not precleaned
        output: list[str] = []
        weighted_probability = 0.0
        weighted_margin = 0.0
        weight_total = 0.0
        changed = premerged
        corrected_count = 0
        known_token_count = 0
        for chunk in chunks:
            internal_noise = re.fullmatch(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})[.']([A-Za-zÀ-ÖØ-öø-ÿ]{2,})", chunk)
            if internal_noise:
                chunk = internal_noise.group(1) + internal_noise.group(2)
                changed = True
                structural_safe = False
            match = _TOKEN_WITH_AFFIX_RE.match(chunk)
            if not match:
                output.append(chunk)
                continue
            prefix, core, suffix = match.groups()
            normalized = core.casefold()
            if normalized in self._token_surfaces:
                known_token_count += 1
                # Exact known tokens must be resolved before connector segmentation.
                # Otherwise terms such as ``allargando`` can be damaged into
                # ``al largando`` merely because both pieces happen to be known.
                if not self._affix_allowed(normalized, prefix, suffix):
                    structural_safe = False
                clean_prefix = "" if prefix in {".", ",", "'"} else prefix
                clean_chunk = clean_prefix + core + suffix
                if clean_chunk != chunk:
                    changed = True
                    structural_safe = False
                output.append(clean_chunk)
                weighted_probability += len(core) * 0.995
                weighted_margin += len(core) * 0.50
                weight_total += len(core)
                continue

            # Prefer a near-exact whole-token correction before trying connector
            # segmentation.  Otherwise OCR such as ``incalAndo`` can be damaged into
            # ``in calando`` merely because both split pieces are valid words.
            early_ranked: list[tuple[float, str]] = []
            for candidate_norm in self._token_shortlist(core, limit=12):
                surface, frequency = self._token_surfaces[candidate_norm]
                if not self._digit_pattern_compatible(core, surface):
                    continue
                early_ranked.append((self._probability(core, surface, frequency), candidate_norm))
            early_ranked.sort(reverse=True)
            if early_ranked:
                close_candidates = [
                    (probability, candidate_norm)
                    for probability, candidate_norm in early_ranked
                    if _levenshtein(normalized, candidate_norm) <= 1
                ]
                close_candidates.sort(reverse=True)
                one_letter_connector_join = (
                    len(normalized) >= 3
                    and normalized[0] in {"a", "e"}
                    and normalized[1:] in self._token_surfaces
                )
                if close_candidates and not one_letter_connector_join:
                    early_probability, early_norm = close_candidates[0]
                    early_second = close_candidates[1][0] if len(close_candidates) > 1 else 0.0
                    if early_probability >= 0.97:
                        surface = self._token_surfaces[early_norm][0]
                        surface = surface[:1].upper() + surface[1:] if core[:1].isupper() else surface.casefold()
                        if not self._affix_allowed(early_norm, prefix, suffix):
                            structural_safe = False
                        clean_prefix = "" if prefix in {".", ",", "'"} else prefix
                        output.append(clean_prefix + surface + suffix)
                        changed = True
                        corrected_count += 1
                        known_token_count += 1
                        weighted_probability += len(core) * early_probability
                        weighted_margin += len(core) * max(0.05, early_probability - early_second)
                        weight_total += len(core)
                        continue

            connector_tokens = _CONNECTOR_TOKENS
            segmented: tuple[str, str] | None = None
            for split_at in range(1, len(normalized)):
                left, right = normalized[:split_at], normalized[split_at:]
                if left in connector_tokens and right in self._token_surfaces:
                    segmented = (left, self._token_surfaces[right][0].casefold())
                    break
                if left in self._token_surfaces and right in {"a", "e"}:
                    segmented = (self._token_surfaces[left][0].casefold(), right)
                    break
            if segmented is None and len(normalized) >= 8:
                # OCR frequently removes a space between adjacent musical words.
                # Split only when *both* sides are established non-trivial tokens;
                # exact lexicon tokens were already handled above, so this cannot
                # damage words such as ``allargando``.
                general_splits: list[tuple[int, str, str]] = []
                for split_at in range(3, len(normalized) - 2):
                    left, right = normalized[:split_at], normalized[split_at:]
                    if left in self._token_surfaces and right in self._token_surfaces:
                        balance = min(len(left), len(right))
                        general_splits.append((balance, left, right))
                if general_splits:
                    _, left, right = max(general_splits)
                    segmented = (
                        self._token_surfaces[left][0].casefold(),
                        self._token_surfaces[right][0].casefold(),
                    )
            if segmented is not None:
                value = " ".join(segmented)
                if core[:1].isupper():
                    value = value[:1].upper() + value[1:]
                if suffix and not self._affix_allowed(segmented[-1], "", suffix):
                    structural_safe = False
                clean_prefix = "" if prefix in {".", ",", "'"} else prefix
                if clean_prefix != prefix:
                    structural_safe = False
                output.append(clean_prefix + value + suffix)
                changed = True
                corrected_count += 1
                known_token_count += 2
                # Splitting one OCR token into two known tokens preserves every letter.
                # It remains safe only when both segments are at least two characters,
                # or one is a recognised grammatical connector.
                structural_safe = structural_safe and (
                    min(len(segmented[0]), len(segmented[1])) >= 2
                    or segmented[0] in connector_tokens
                    or segmented[1] in connector_tokens
                )
                weighted_probability += len(core) * 0.985
                weighted_margin += len(core) * 0.40
                weight_total += len(core)
                continue
            if len(core) <= 2:
                output.append(chunk)
                continue
            ranked: list[tuple[float, str]] = []
            for candidate_norm in self._token_shortlist(core):
                surface, frequency = self._token_surfaces[candidate_norm]
                if not self._digit_pattern_compatible(core, surface):
                    continue
                ranked.append((self._probability(core, surface, frequency), candidate_norm))
            ranked.sort(reverse=True)
            if not ranked:
                output.append(chunk)
                continue
            probability, best_norm = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = probability - second
            edit_ratio = _levenshtein(normalized, best_norm) / max(len(normalized), len(best_norm), 1)
            if probability >= 0.79 and margin >= 0.055 and edit_ratio <= 0.45:
                surface = self._token_surfaces[best_norm][0]
                if core[:1].isupper() and surface:
                    surface = surface[:1].upper() + surface[1:]
                else:
                    surface = surface.casefold()
                if not self._affix_allowed(best_norm, prefix, suffix):
                    structural_safe = False
                clean_prefix = "" if prefix in {".", ",", "'"} else prefix
                output.append(clean_prefix + surface + suffix)
                changed = True
                corrected_count += 1
                known_token_count += 1
                weighted_probability += len(core) * probability
                weighted_margin += len(core) * margin
                weight_total += len(core)
            else:
                output.append(chunk)
        required_known = 1 if premerged and len(chunks) == 1 else max(2, len(chunks) - 1)
        if weight_total <= 0 or known_token_count < required_known:
            return DirectionSuggestion(observed, 0.0, 0.0, False, "token", False, 1.0)
        canonical_output: list[str] = []
        abbreviation_changed = False
        for item in output:
            key = normalize_direction(item).rstrip(".")
            if key in _ABBREVIATION_MAP and not item.rstrip().endswith("."):
                item = _ABBREVIATION_MAP[key]
                changed = True
                abbreviation_changed = True
            canonical_output.append(item)
        suggestion = " ".join(canonical_output)
        probability = weighted_probability / weight_total
        margin = weighted_margin / weight_total
        changed_value = normalize_direction(suggestion) != normalize_direction(observed)
        observed_word_count = len(_WORD_RE.findall(observed))
        suggested_word_count = len(_WORD_RE.findall(suggestion))
        edit_ratio = _levenshtein(normalize_direction(observed), normalize_direction(suggestion)) / max(
            len(normalize_direction(observed)), len(normalize_direction(suggestion)), 1
        )
        # Automatic write-back is deliberately narrower than suggestion generation.
        # Every textual token must be understood. Character-preserving insertion or
        # removal of whitespace is safe when the split decoder has not crossed a known
        # token boundary; all other structural changes remain review-only.
        compact_observed = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9]", "", observed).casefold()
        compact_suggestion = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9]", "", suggestion).casefold()
        character_preserving = compact_observed == compact_suggestion
        output_tokens = [token.casefold() for token in _WORD_RE.findall(suggestion)]
        all_understood = all(token in self._token_surfaces or token in _CONNECTOR_TOKENS for token in output_tokens)
        dangling_connector = bool(output_tokens) and output_tokens[-1] in _CONNECTOR_TOKENS
        unsafe_singletons = [
            token
            for token in output_tokens
            if len(token) == 1
            and token not in {"a", "e"}
            and not (len(output_tokens) == 1 and token in _SINGLE_TOKEN_DYNAMICS)
        ]
        digit_pattern_preserved = self._digit_pattern_compatible(observed, suggestion)
        exact_phrase = normalize_direction(suggestion) in self.frequencies
        if (
            not changed_value
            and all_understood
            and not dangling_connector
            and not unsafe_singletons
        ):
            return DirectionSuggestion(observed, probability, margin, False, "token-preserve", False, 0.0)
        safe_structure = structural_safe and (
            observed_word_count == suggested_word_count or character_preserving
        )
        safe = (
            changed_value
            and (corrected_count > 0 or premerged)
            and all_understood
            and not unsafe_singletons
            and digit_pattern_preserved
            and safe_structure
            and (not dangling_connector or exact_phrase)
            and not (abbreviation_changed and corrected_count > 0)
            and edit_ratio <= 0.22
            and probability >= 0.985
            and margin >= 0.24
        )
        has_affixed_token = any(
            (match := _TOKEN_WITH_AFFIX_RE.match(chunk)) is not None
            and bool(match.group(1) or match.group(3))
            for chunk in suggestion.split()
        )
        if not safe:
            safety_class = "none"
        elif character_preserving:
            safety_class = "character-preserving"
        elif exact_phrase:
            safety_class = "known-phrase"
        elif corrected_count == 1 and not has_affixed_token:
            safety_class = "novel-single-token"
        else:
            # Multi-token corrections in an unseen composition can hide a missing
            # connector (for example ``cantabilee olce``). They remain review-only.
            safety_class = "none"
        return DirectionSuggestion(
            suggestion,
            probability,
            margin,
            changed_value,
            "token",
            safe and safety_class != "none",
            edit_ratio,
            safety_class,
        )

    @lru_cache(maxsize=4096)
    def suggest(self, observed: str) -> DirectionSuggestion:
        normalized = normalize_direction(observed)
        if not normalized:
            return DirectionSuggestion(observed, 0.0, 0.0, False, "empty", False, 0.0)
        if not self.model_enabled:
            return DirectionSuggestion(observed, 0.0, 0.0, False, "disabled", False, 0.0)
        # A terminal full stop is conventional in many printed tempo headings
        # (``Allegretto.``) and is part of the source text, not an OCR error.  Preserve
        # exactly one terminal period when the unpunctuated phrase is itself a known
        # lexicon entry.  Internal punctuation and repeated punctuation deliberately do
        # not pass this gate.
        stripped = observed.rstrip()
        if stripped.endswith(".") and not stripped.endswith(".."):
            unpunctuated = stripped[:-1].rstrip()
            unpunctuated_key = normalize_direction(unpunctuated)
            if unpunctuated_key in self.frequencies:
                evidence = self._direct_suggest(unpunctuated)
                return DirectionSuggestion(
                    observed,
                    evidence.probability,
                    evidence.margin,
                    False,
                    "terminal-punctuation-preserve",
                    False,
                    0.0,
                    "source-punctuation",
                )
        direct = self._direct_suggest(observed)
        compositional = self._compositional_suggest(observed)
        if normalized in self.frequencies:
            return direct
        # A novel phrase made entirely from known musical tokens is stronger evidence
        # than a forced whole-phrase lexicon match.  Preserve the observed text when
        # the compositional decoder understood it without changing any normalized
        # token.  This is especially important for previously unseen combinations such
        # as ``Andante dolce ma espressivo``.
        if not compositional.changed and compositional.method == "token-preserve":
            return compositional
        if not compositional.changed:
            return direct
        # A first compositional pass may canonicalise an abbreviation and thereby make
        # a second phrase-level correction unambiguous (for example ``rall a fempo`` ->
        # ``rall. a tempo``).  Keep this as a review suggestion unless the original
        # unattended-writeback gate was already satisfied.
        second_pass = self._direct_suggest(compositional.text)
        if second_pass.changed and second_pass.probability >= 0.90 and second_pass.margin >= 0.08:
            combined_edit = _levenshtein(normalized, normalize_direction(second_pass.text)) / max(
                len(normalized), len(normalize_direction(second_pass.text)), 1
            )
            if combined_edit <= 0.28:
                compositional = DirectionSuggestion(
                    second_pass.text,
                    min(compositional.probability, second_pass.probability),
                    min(compositional.margin, second_pass.margin),
                    True,
                    "hybrid",
                    False,
                    combined_edit,
                )
        compositional_tokens = [token.casefold() for token in _WORD_RE.findall(compositional.text)]
        connectors = {"a", "e", "et", "and", "und", "ma", "al", "da", "de", "in", "au", "en", "un", "con"}
        dangling_connector = bool(compositional_tokens) and compositional_tokens[-1] in connectors
        if dangling_connector and not compositional.autocorrect_safe and normalize_direction(compositional.text) not in self.frequencies:
            return direct

        direct_tokens = len(_WORD_RE.findall(direct.text))
        observed_tokens = len(_WORD_RE.findall(observed))
        direct_shape_mismatch = abs(direct_tokens - observed_tokens) >= 2

        # A character-preserving token reconstruction is stronger evidence than an
        # unsafe whole-phrase guess, especially when the latter changes substantially
        # more of the observed text.
        if (
            compositional.autocorrect_safe
            and not direct.autocorrect_safe
            and compositional.edit_ratio + 0.04 < direct.edit_ratio
            and compositional.probability >= 0.985
        ):
            return compositional

        # Known complete phrases are more constrained than token-wise reconstruction.
        # Prefer them unless the candidate clearly changes the observed phrase shape or
        # compositional evidence is decisively stronger.
        if direct.probability >= 0.84 and direct.margin >= 0.045 and not direct_shape_mismatch:
            if not (
                compositional.probability >= direct.probability + 0.10
                and compositional.margin >= direct.margin + 0.06
                and compositional.edit_ratio <= direct.edit_ratio
            ):
                return direct
        if compositional.probability >= 0.88 and compositional.margin >= 0.09:
            return compositional
        return direct

    def should_autocorrect(self, suggestion: DirectionSuggestion) -> bool:
        """Return whether the suggestion is safe for unattended MusicXML write-back.

        Recall is intentionally sacrificed here.  Suggestions that involve uncertain
        word splitting, token loss, or unknown terms are still shown to the user but are
        never applied silently.
        """
        if not suggestion.changed or not suggestion.autocorrect_safe:
            return False
        if suggestion.method == "token":
            # Character-preserving joins are structurally different from probabilistic
            # spelling changes.  Earlier code inferred this class from a probability
            # interval, which could accidentally admit an ordinary ambiguous token.
            if suggestion.safety_class == "character-preserving":
                return (
                    suggestion.probability >= 0.984
                    and suggestion.margin >= 0.35
                    and suggestion.edit_ratio <= 0.16
                )
            if suggestion.safety_class == "known-phrase":
                return (
                    suggestion.probability >= 0.990
                    and suggestion.margin >= 0.30
                    and suggestion.edit_ratio <= 0.18
                    and normalize_direction(suggestion.text) in self.frequencies
                )
            if suggestion.safety_class == "novel-single-token":
                return (
                    suggestion.probability > 0.995
                    and suggestion.margin >= 0.35
                    and suggestion.edit_ratio <= 0.16
                )
            return False
        return suggestion.probability > 0.995 and suggestion.margin >= 0.30 and suggestion.edit_ratio <= 0.20



def iter_known_terms(corrector: DirectionCorrector) -> Iterable[str]:
    yield from corrector.lexicon
