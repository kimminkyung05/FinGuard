"""Video-risk policy, deliberately separated from analysis confidence."""

# Validate and tune these defaults on a representative dataset before release.
MINIMUM_CONFIDENCE = 0.50
RETRY_THRESHOLD = 0.40
BLOCK_THRESHOLD = 0.75


def _bounded(value):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def get_thresholds():
    """Return the policy thresholds included in every decision payload."""
    return {
        "minimum_confidence": MINIMUM_CONFIDENCE,
        "retry_threshold": RETRY_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD,
    }


def assess_risk(video_fake_score, confidence_score, **_unused_quality_signals):
    """Make a video-only decision without using confidence to inflate risk.

    Future audio and conversation scores can be fused into ``risk_score`` in a
    dedicated multimodal policy. Until then, it equals ``video_fake_score``.
    """
    risk_score = _bounded(video_fake_score)
    confidence_score = _bounded(confidence_score)
    if confidence_score < MINIMUM_CONFIDENCE:
        return {
            "risk_score": risk_score,
            "risk_level": "UNDETERMINED",
            "decision": "RETRY",
            "reason_codes": ["LOW_CONFIDENCE"],
        }
    if risk_score >= BLOCK_THRESHOLD:
        level, decision, reason = "HIGH", "BLOCK", "VIDEO_DEEPFAKE_HIGH"
    elif risk_score >= RETRY_THRESHOLD:
        level, decision, reason = "MEDIUM", "RETRY", "VIDEO_DEEPFAKE_UNCERTAIN"
    else:
        level, decision, reason = "LOW", "APPROVE", "LOW_RISK"
    return {
        "risk_score": risk_score,
        "risk_level": level,
        "decision": decision,
        "reason_codes": [reason],
    }
