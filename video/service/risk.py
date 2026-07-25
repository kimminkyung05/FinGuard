"""Financial-service policy decisions derived from model and quality signals."""

LOW_RISK_THRESHOLD = 30
HIGH_RISK_THRESHOLD = 70
MINIMUM_CONFIDENCE_FOR_DECISION = 60
LOW_FACE_DETECTION_RATIO = 0.60
MINIMUM_VALID_FRAMES = 30
HIGH_BLUR_SCORE = 0.40
ABNORMAL_BRIGHTNESS_LOW = 0.20
ABNORMAL_BRIGHTNESS_HIGH = 0.80


def assess_risk(video_fake_score, confidence_score, face_detection_ratio=None,
                valid_frames=None, blur_score=None, brightness_score=None,
                prediction_std=None):
    """Apply explainable financial-service decision rules.

    Low confidence always produces RETRY, preventing automatic approval from a
    poorly observed video.  Otherwise the risk score follows fake probability.
    """
    fake_score = min(1.0, max(0.0, float(video_fake_score or 0.0)))
    confidence_score = min(100.0, max(0.0, float(confidence_score or 0.0)))
    risk_score = int(round(fake_score * 100))
    reason_codes = []
    reasons = []

    if risk_score >= HIGH_RISK_THRESHOLD:
        reason_codes.append("HIGH_FAKE_PROBABILITY")
        reasons.append("The video has a high predicted fake probability.")
    elif risk_score >= LOW_RISK_THRESHOLD:
        reason_codes.append("MODERATE_FAKE_PROBABILITY")
        reasons.append("The video has a moderate predicted fake probability.")

    if confidence_score < MINIMUM_CONFIDENCE_FOR_DECISION:
        reason_codes.append("LOW_CONFIDENCE")
        reasons.append("The analysis confidence is too low for automatic approval.")
    if face_detection_ratio is not None and face_detection_ratio < LOW_FACE_DETECTION_RATIO:
        reason_codes.append("LOW_FACE_DETECTION_RATE")
        reasons.append("Too few frames contained a detected face.")
    if valid_frames is not None and valid_frames < MINIMUM_VALID_FRAMES:
        reason_codes.append("INSUFFICIENT_VALID_FRAMES")
        reasons.append("Too few valid frames were available.")
    if prediction_std is not None and prediction_std > 0.30:
        reason_codes.append("HIGH_PREDICTION_VARIANCE")
        reasons.append("Frame-level predictions varied substantially.")
    if blur_score is not None and blur_score > HIGH_BLUR_SCORE:
        reason_codes.append("BLURRY_VIDEO")
        reasons.append("The video is excessively blurry.")
    if brightness_score is not None and (brightness_score < ABNORMAL_BRIGHTNESS_LOW or brightness_score > ABNORMAL_BRIGHTNESS_HIGH):
        reason_codes.append("ABNORMAL_BRIGHTNESS")
        reasons.append("The video brightness is abnormal.")

    if confidence_score < MINIMUM_CONFIDENCE_FOR_DECISION:
        decision = "RETRY"
    elif risk_score < LOW_RISK_THRESHOLD:
        decision = "APPROVE"
    elif risk_score < HIGH_RISK_THRESHOLD:
        decision = "RETRY"
    else:
        decision = "BLOCK"

    if risk_score < LOW_RISK_THRESHOLD:
        risk_level = "LOW"
    elif risk_score < HIGH_RISK_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": decision,
        "reason_codes": reason_codes,
        "reasons": reasons,
    }
