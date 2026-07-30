"""Explainable aggregation of frame-level fake probabilities."""

from __future__ import division

import math


# Tune these values against a validated dataset before production use.
HIGH_RISK_FRAME_THRESHOLD = 0.5
TOP_K_RATIO = 0.10


def _valid_scores(frame_scores):
    """Return finite frame probabilities, clamped to the [0, 1] range."""
    scores = []
    for score in frame_scores or []:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            scores.append(min(1.0, max(0.0, value)))
    return scores


def aggregate_frame_scores(frame_scores, high_risk_threshold=HIGH_RISK_FRAME_THRESHOLD):
    """Summarize frame probabilities without changing the video risk meaning.

    ``video_fake_score`` is the mean fake probability and remains in [0, 1].
    The other values make the aggregation auditable and can later be shared by
    video, audio, and multimodal fusion layers.
    """
    scores = sorted(_valid_scores(frame_scores))
    empty_statistics = {
        "mean_score": 0.0,
        "max_score": 0.0,
        "min_score": 0.0,
        "std_score": 0.0,
        "top_k_mean": 0.0,
        "high_risk_frame_ratio": 0.0,
        "valid_frame_count": 0,
    }
    if not scores:
        return {"video_fake_score": 0.0, "statistics": empty_statistics}

    count = len(scores)
    mean_score = sum(scores) / count
    top_k_count = max(1, int(math.ceil(count * TOP_K_RATIO)))
    top_k_mean = sum(scores[-top_k_count:]) / top_k_count
    std_score = math.sqrt(sum((score - mean_score) ** 2 for score in scores) / count)
    high_risk_ratio = sum(score >= high_risk_threshold for score in scores) / float(count)
    statistics = {
        "mean_score": mean_score,
        "max_score": scores[-1],
        "min_score": scores[0],
        "std_score": std_score,
        "top_k_mean": top_k_mean,
        "high_risk_frame_ratio": high_risk_ratio,
        "valid_frame_count": count,
    }
    return {"video_fake_score": mean_score, "statistics": statistics}
