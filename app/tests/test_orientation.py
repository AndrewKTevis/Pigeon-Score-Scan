from pathlib import Path

import cv2
import numpy as np

from scorescan.models import PageInfo
from scorescan.imaging import preprocess_page
from scorescan.orientation import (
    FEATURE_NAMES,
    PageOrientationClassifier,
    extract_orientation_features,
    rotate_quadrant,
)


def synthetic_score() -> np.ndarray:
    image = np.full((1000, 700), 255, dtype=np.uint8)
    for system in range(6):
        top = 100 + system * 140
        for line in range(5):
            cv2.line(image, (50, top + line * 12), (650, top + line * 12), 0, 1)
        cv2.ellipse(image, (70, top + 24), (10, 28), 0, 0, 360, 0, 3)
        cv2.circle(image, (75, top + 12), 5, 0, -1)
        for x in range(140, 620, 60):
            y = top + 24
            cv2.ellipse(image, (x, y), (6, 4), -20, 0, 360, 0, -1)
            cv2.line(image, (x + 5, y), (x + 5, y - 30), 0, 1)
        cv2.line(image, (650, top), (650, top + 48), 0, 2)
    cv2.putText(image, "Allegretto", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
    return image


def test_orientation_features_preserve_staff_line_axis() -> None:
    upright = synthetic_score()
    rotated = rotate_quadrant(upright, 90)
    upright_features = dict(zip(FEATURE_NAMES, extract_orientation_features(upright)))
    rotated_features = dict(zip(FEATURE_NAMES, extract_orientation_features(rotated)))
    assert len(upright_features) == len(FEATURE_NAMES)
    assert upright_features["horizontal_line_density"] > upright_features["vertical_line_density"]
    assert rotated_features["vertical_line_density"] > rotated_features["horizontal_line_density"]


def test_orientation_classifier_is_selective_and_never_auto_flips_180() -> None:
    classifier = PageOrientationClassifier()
    assert classifier.model_status == "verified"
    upright = synthetic_score()
    assert classifier.classify(upright).degrees == 0
    clockwise = classifier.classify(rotate_quadrant(upright, 90))
    assert clockwise.degrees == 270
    assert clockwise.applied
    upside_down = classifier.classify(rotate_quadrant(upright, 180))
    assert upside_down.degrees == 180
    assert not upside_down.applied


def test_preprocess_never_applies_quarter_turn(tmp_path: Path) -> None:
    source = tmp_path / "rotated.png"
    rotated = rotate_quadrant(synthetic_score(), 90)
    cv2.imwrite(str(source), rotated)
    page = PageInfo(1, source.name, str(source), width=rotated.shape[1], height=rotated.shape[0])
    preprocess_page(page, tmp_path / "normalised")
    normalized = cv2.imread(str(page.normalized_path), cv2.IMREAD_GRAYSCALE)
    assert page.orientation_degrees == 0
    assert not page.orientation_applied
    assert page.orientation_probability is None
    assert page.orientation_margin is None
    assert page.orientation_model_version is None
    assert page.orientation_model_status == "automatic-rotation-disabled"
    assert page.orientation_probabilities == {}
    assert (page.width, page.height) == (rotated.shape[1], rotated.shape[0])
    assert normalized is not None
    assert normalized.shape == rotated.shape


def test_preprocess_never_applies_small_angle_deskew(tmp_path: Path) -> None:
    source = tmp_path / "skewed.png"
    upright = synthetic_score()
    height, width = upright.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 2.0, 1.0)
    skewed = cv2.warpAffine(
        upright,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    cv2.imwrite(str(source), skewed)
    page = PageInfo(1, source.name, str(source), width=width, height=height)
    preprocess_page(page, tmp_path / "normalised")
    normalized = cv2.imread(str(page.normalized_path), cv2.IMREAD_GRAYSCALE)
    assert normalized is not None
    assert normalized.shape == skewed.shape
    assert page.orientation_degrees == 0
    assert not page.orientation_applied
