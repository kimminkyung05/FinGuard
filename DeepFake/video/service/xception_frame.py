"""Shared single-frame Xception predictor used by upload and real-time paths."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from video.classification.detect_from_video import (
    FAKE_CLASS_INDEX,
    crop_face,
    detect_largest_face,
    preprocess_image,
)


def predict_xception_frame(frame_bgr: np.ndarray, model: Any, device: torch.device) -> dict[str, Any]:
    bbox = detect_largest_face(frame_bgr)
    if bbox is None:
        return {"face_detected": False, "bbox": None}

    face = crop_face(frame_bgr, bbox, margin=1.3)
    if face.size == 0:
        return {"face_detected": False, "bbox": bbox}

    tensor = preprocess_image(face, cuda=device.type == "cuda", device=device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    fake_score = float(probabilities[FAKE_CLASS_INDEX])
    real_score = float(probabilities[1 - FAKE_CLASS_INDEX])
    return {
        "face_detected": True,
        "bbox": tuple(int(value) for value in bbox),
        "raw_output": logits[0].detach().cpu().tolist(),
        "real_score": real_score,
        "fake_score": fake_score,
        "tensor_min": float(tensor.min().item()),
        "tensor_max": float(tensor.max().item()),
        "tensor_mean": float(tensor.mean().item()),
        "face": face,
    }
