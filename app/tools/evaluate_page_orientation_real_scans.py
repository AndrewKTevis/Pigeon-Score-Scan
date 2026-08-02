from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from scorescan.orientation import ORIENTATIONS, PageOrientationClassifier, rotate_quadrant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    classifier = PageOrientationClassifier()
    rows: list[dict[str, object]] = []
    correct = 0
    rotated_cases = 0
    auto_applied = 0
    correct_auto_applied = 0
    harmful_actions = 0
    upright_false_rotations = 0
    for image_path in args.image:
        upright = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if upright is None:
            continue
        for source_rotation in ORIENTATIONS:
            rotated = rotate_quadrant(upright, source_rotation)
            expected = (-source_rotation) % 360
            result = classifier.classify(rotated)
            predicted_correct = result.degrees == expected
            correct += int(predicted_correct)
            if expected != 0:
                rotated_cases += 1
            if result.applied:
                auto_applied += 1
                correct_auto_applied += int(predicted_correct)
                harmful_actions += int(not predicted_correct)
                upright_false_rotations += int(expected == 0)
            rows.append({
                'image': str(image_path),
                'source_rotation': source_rotation,
                'expected_correction': expected,
                'predicted_correction': result.degrees,
                'probability': result.probability,
                'margin': result.margin,
                'applied': result.applied,
                'correct_class': predicted_correct,
                'harmful_action': bool(result.applied and not predicted_correct),
                'probabilities': {str(key): value for key, value in result.probabilities.items()},
            })
    report = {
        'model': classifier.model_version,
        'model_status': classifier.model_status,
        'cases': len(rows),
        'classification_accuracy': correct / max(len(rows), 1),
        'rotated_cases': rotated_cases,
        'auto_applied': auto_applied,
        'auto_correction_precision': correct_auto_applied / max(auto_applied, 1),
        'auto_correction_coverage': correct_auto_applied / max(rotated_cases, 1),
        'harmful_actions': harmful_actions,
        'upright_false_rotations': upright_false_rotations,
        'rows': rows,
        'scope': 'two supplied printed-score pages rotated by exact quadrants; orientation only, not end-to-end OMR accuracy',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
