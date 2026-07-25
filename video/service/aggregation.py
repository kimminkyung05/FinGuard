"""Explainable aggregation of frame-level fake probabilities."""

from __future__ import division

import math
from typing import Dict, List


HIGH_RISK_FRAME_THRESHOLD = 0.70
HIGH_RISK_RATIO_BONUS_THRESHOLD = 0.60
HIGH_RISK_RATIO_BONUS = 0.05


def _valid_scores(frame_scores):
    """Return finite scores clamped to the probability range."""
    valid = []
    for score in frame_scores or []:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid.append(min(1.0, max(0.0, value)))
    return valid


def aggregate_frame_scores(frame_scores):
    """Aggregate frame probabilities into explainable video-level statistics.

    The base video score is the mean frame score.  A small, documented bonus is
    applied only when most frames are high risk; no additional model is used.
    """
    scores = _valid_scores(frame_scores)
    if not scores:
        return {
            "mean_score": 0.0,
            "median_score": 0.0,
            "max_score": 0.0,
            "score_std": 0.0,
            "high_risk_frame_ratio": 0.0,
            "video_fake_score": 0.0,
            "valid_score_count": 0,
        }

    scores.sort()
    count = len(scores)
    mean_score = sum(scores) / count
    midpoint = count // 2
    if count % 2:
        median_score = scores[midpoint]
    else:
        median_score = (scores[midpoint - 1] + scores[midpoint]) / 2.0
    score_std = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / count)
    high_risk_ratio = sum(score >= HIGH_RISK_FRAME_THRESHOLD for score in scores) / float(count)

    video_fake_score = mean_score
    if high_risk_ratio >= HIGH_RISK_RATIO_BONUS_THRESHOLD:
        video_fake_score = min(1.0, mean_score + HIGH_RISK_RATIO_BONUS)

    return {
        "mean_score": mean_score,
        "median_score": median_score,
        "max_score": scores[-1],
        "score_std": score_std,
        "high_risk_frame_ratio": high_risk_ratio,
        "video_fake_score": video_fake_score,
        "valid_score_count": count,
    }
