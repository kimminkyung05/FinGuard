"""JSON-safe, modality-ready output contract for inference consumers."""

import json
import os
import uuid
from datetime import datetime, timezone

from .risk import get_thresholds


def _number(value, default=0.0):
    try:
        value = value.item()
    except AttributeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_output(aggregation, confidence, risk, total_frames, valid_frames,
                 face_detected_frames, blur_score=None, brightness_score=None,
                 processing_time_ms=0, model_name="Xception", model_version="v1.0",
                 input_path=None, annotated_video_path=None, status="SUCCESS",
                 source_total_frames=None):
    """Build a stable video result that can later carry audio/fusion scores."""
    video_fake_score = _number(aggregation.get("video_fake_score"))
    confidence_score = _number(confidence.get("confidence_score"))
    risk_score = _number(risk.get("risk_score"))
    statistics = dict(aggregation.get("statistics", {}))
    components = dict(confidence.get("components", {}))
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {
            "input_type": "video",
            "input_path": input_path,
            "file_name": os.path.basename(input_path) if input_path else None,
        },
        "scores": {
            "video_fake_score": video_fake_score,
            "video_fake_percent": video_fake_score * 100.0,
            "confidence_score": confidence_score,
            "confidence_percent": confidence_score * 100.0,
            # Reserved for future audio and multimodal score producers.
            "audio_fake_score": None,
            "conversation_risk_score": None,
        },
        "risk": {
            "risk_score": risk_score,
            "risk_percent": risk_score * 100.0,
            "risk_level": risk.get("risk_level", "UNDETERMINED"),
            "decision": risk.get("decision", "RETRY"),
            "reason_codes": list(risk.get("reason_codes", [])),
        },
        "frame_analysis": {
            "total_frames": _integer(source_total_frames, _integer(total_frames)),
            "analyzed_frames": _integer(total_frames),
            "valid_face_frames": _integer(valid_frames),
            "statistics": {key: _number(value) for key, value in statistics.items()},
        },
        "quality": {
            "components": {key: _number(value) for key, value in components.items()},
            "quality_flags": list(confidence.get("quality_flags", [])),
        },
        "thresholds": get_thresholds(),
        "model_info": {"model_name": model_name, "model_version": model_version},
        "processing": {"processing_time_ms": _integer(processing_time_ms), "status": status},
        "artifacts": {"annotated_video_path": annotated_video_path},
    }


def dumps_output(payload):
    """Serialize a payload created by :func:`build_output` to JSON."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
