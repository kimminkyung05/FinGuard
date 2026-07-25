"""JSON-safe output contract for financial-service consumers."""

import json
import uuid
from datetime import datetime, timezone


def _number(value, default=0.0):
    """Convert numeric-like values, including NumPy scalars, to Python values."""
    try:
        converted = value.item()
    except AttributeError:
        converted = value
    try:
        return float(converted)
    except (TypeError, ValueError):
        return default


def build_output(aggregation, confidence, risk, total_frames, valid_frames,
                 face_detected_frames, blur_score, brightness_score,
                 processing_time_ms, model_name="Xception", model_version="1.0.0"):
    """Build the stable, serializable result payload."""
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "result": {
            "risk_score": int(risk["risk_score"]),
            "risk_level": risk["risk_level"],
            "confidence_score": _number(confidence["confidence_score"]),
            "confidence_level": confidence["confidence_level"],
            "decision": risk["decision"],
        },
        "analysis": {
            "video_fake_score": _number(aggregation["video_fake_score"]),
            "mean_frame_score": _number(aggregation["mean_score"]),
            "median_frame_score": _number(aggregation["median_score"]),
            "max_frame_score": _number(aggregation["max_score"]),
            "score_std": _number(aggregation["score_std"]),
            "high_risk_frame_ratio": _number(aggregation["high_risk_frame_ratio"]),
        },
        "quality": {
            "total_frames": int(total_frames),
            "valid_frames": int(valid_frames),
            "face_detected_frames": int(face_detected_frames),
            "face_detection_ratio": _number(confidence["face_detection_ratio"]),
            "blur_score": _number(blur_score),
            "brightness_score": _number(brightness_score),
        },
        "explanation": {
            "reason_codes": list(risk["reason_codes"]),
            "reasons": list(risk["reasons"]),
            "confidence_reasons": list(confidence["confidence_reasons"]),
        },
        "processing": {"processing_time_ms": int(processing_time_ms)},
        "model": {"name": model_name, "version": model_version},
    }


def dumps_output(payload):
    """Serialize a payload created by :func:`build_output` to JSON."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)
