"""Confidence scoring based on video quality and prediction consistency."""

from __future__ import division

import math


MINIMUM_FRAME_COUNT = 30
LOW_FACE_DETECTION_RATE = 0.60
LOW_VALID_FRAME_RATIO = 0.60
MIN_SHARPNESS_SCORE = 0.60
TOO_DARK_THRESHOLD = 0.20
TOO_BRIGHT_THRESHOLD = 0.80
MAX_PREDICTION_STD = 0.30
MIN_PREDICTION_CONSISTENCY = 0.60


def _bounded(value, default=0.0):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _ratio(numerator, denominator):
    return 0.0 if denominator <= 0 else _bounded(float(numerator) / denominator)


def _prediction_consistency(frame_scores):
    values = []
    for score in frame_scores or []:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(_bounded(value))
    if len(values) < 2:
        return 0.0
    mean_score = sum(values) / len(values)
    std_score = math.sqrt(sum((value - mean_score) ** 2 for value in values) / len(values))
    return _bounded(1.0 - (std_score / MAX_PREDICTION_STD))


def calculate_confidence(total_frames, valid_frames, face_detected_frames,
                         frame_scores, blur_score, brightness_score):
    """Return an independent, normalized analysis-confidence assessment.

    ``blur_score`` is 0 for sharp and 1 for blurry; therefore the returned
    ``sharpness_score`` is its inverse. Confidence never changes fake risk.
    """
    total_frames = max(0, int(total_frames or 0))
    valid_frames = min(total_frames, max(0, int(valid_frames or 0)))
    face_detected_frames = min(total_frames, max(0, int(face_detected_frames or 0)))
    components = {
        "face_detection_rate": _ratio(face_detected_frames, total_frames),
        "valid_frame_ratio": _ratio(valid_frames, total_frames),
        "sharpness_score": 1.0 - _bounded(blur_score),
        "brightness_score": _bounded(brightness_score),
        "prediction_consistency": _prediction_consistency(frame_scores),
    }
    frame_count_score = _bounded(total_frames / float(MINIMUM_FRAME_COUNT))
    confidence_score = sum((
        components["face_detection_rate"] * 0.25,
        components["valid_frame_ratio"] * 0.20,
        frame_count_score * 0.15,
        components["sharpness_score"] * 0.15,
        (1.0 - abs(components["brightness_score"] - 0.5) / 0.5) * 0.10,
        components["prediction_consistency"] * 0.15,
    ))
    flags = []
    if components["face_detection_rate"] < LOW_FACE_DETECTION_RATE:
        flags.append("LOW_FACE_DETECTION_RATE")
    if components["valid_frame_ratio"] < LOW_VALID_FRAME_RATIO or total_frames < MINIMUM_FRAME_COUNT:
        flags.append("INSUFFICIENT_VALID_FRAMES")
    if components["sharpness_score"] < MIN_SHARPNESS_SCORE:
        flags.append("BLURRY_VIDEO")
    if components["brightness_score"] < TOO_DARK_THRESHOLD:
        flags.append("TOO_DARK")
    elif components["brightness_score"] > TOO_BRIGHT_THRESHOLD:
        flags.append("TOO_BRIGHT")
    if components["prediction_consistency"] < MIN_PREDICTION_CONSISTENCY:
        flags.append("INCONSISTENT_PREDICTIONS")
    return {
        "confidence_score": _bounded(confidence_score),
        "components": components,
        "quality_flags": flags,
    }
