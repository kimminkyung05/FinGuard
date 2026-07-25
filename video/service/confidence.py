"""Confidence scoring based on video quality and prediction consistency."""

from __future__ import division

import math


MINIMUM_FRAME_COUNT = 30
LOW_CONFIDENCE_THRESHOLD = 60
HIGH_CONFIDENCE_THRESHOLD = 80
MAX_ACCEPTABLE_BLUR = 1.0
IDEAL_BRIGHTNESS = 0.5
MAX_PREDICTION_STD = 0.30


def _ratio(numerator, denominator):
    """Safely calculate a ratio in the range [0, 1]."""
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / denominator))


def _score_std(frame_scores):
    values = []
    for score in frame_scores or []:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(min(1.0, max(0.0, value)))
    if len(values) < 2:
        return None
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def calculate_confidence(total_frames, valid_frames, face_detected_frames,
                         frame_scores, blur_score, brightness_score):
    """Return confidence independently from the model's fake probability.

    Blur is expected on a 0 (sharp) to 1 (very blurry) scale. Brightness is
    expected on a 0 (dark) to 1 (bright) scale, with 0.5 being ideal.
    """
    total_frames = max(0, int(total_frames or 0))
    valid_frames = min(total_frames, max(0, int(valid_frames or 0)))
    face_detected_frames = min(total_frames, max(0, int(face_detected_frames or 0)))
    face_ratio = _ratio(face_detected_frames, total_frames)
    valid_ratio = _ratio(valid_frames, total_frames)
    frame_quality = min(1.0, total_frames / float(MINIMUM_FRAME_COUNT))
    blur = min(MAX_ACCEPTABLE_BLUR, max(0.0, float(blur_score or 0.0)))
    blur_quality = 1.0 - (blur / MAX_ACCEPTABLE_BLUR)
    brightness = min(1.0, max(0.0, float(brightness_score or 0.0)))
    brightness_quality = max(0.0, 1.0 - abs(brightness - IDEAL_BRIGHTNESS) / IDEAL_BRIGHTNESS)
    prediction_std = _score_std(frame_scores)
    consistency_quality = 0.0 if prediction_std is None else max(
        0.0, 1.0 - prediction_std / MAX_PREDICTION_STD)

    confidence_score = round(100.0 * (
        face_ratio * 0.25 + valid_ratio * 0.20 + frame_quality * 0.15 +
        blur_quality * 0.15 + brightness_quality * 0.10 + consistency_quality * 0.15
    ), 2)

    reasons = []
    if total_frames == 0:
        reasons.append("No frames were available for analysis.")
    if face_ratio < 0.60:
        reasons.append("Face detection rate is below 60%.")
    if valid_ratio < 0.60:
        reasons.append("Valid frame rate is below 60%.")
    if total_frames < MINIMUM_FRAME_COUNT:
        reasons.append("Fewer than {0} frames were analysed.".format(MINIMUM_FRAME_COUNT))
    if blur_quality < 0.60:
        reasons.append("Video quality is reduced by blur.")
    if brightness_quality < 0.60:
        reasons.append("Video brightness is outside the preferred range.")
    if prediction_std is None:
        reasons.append("Prediction consistency cannot be measured.")
    elif prediction_std > MAX_PREDICTION_STD:
        reasons.append("Frame predictions are inconsistent.")

    if confidence_score < LOW_CONFIDENCE_THRESHOLD:
        level = "LOW"
    elif confidence_score < HIGH_CONFIDENCE_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return {
        "confidence_score": confidence_score,
        "confidence_level": level,
        "confidence_reasons": reasons,
        "face_detection_ratio": face_ratio,
        "prediction_std": prediction_std,
    }
