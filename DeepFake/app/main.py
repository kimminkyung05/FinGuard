"""Minimal HTTP wrapper around the existing video inference CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "video" / "face_detection" / "xception" / "all_c23.p"
API_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "api"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
INFERENCE_TIMEOUT_SECONDS = 600

app = FastAPI(title="FaceForensics Video Analysis API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _process_error_detail(error: subprocess.CalledProcessError) -> str:
    """Return CLI diagnostics without hiding stdout/stderr from API users."""
    diagnostics = (error.stderr or error.stdout or "Inference process failed.").strip()
    return diagnostics[-4000:]


@app.post("/video/analyze", status_code=status.HTTP_200_OK)
async def analyze_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Run the existing Xception CLI for an uploaded video and return its JSON."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported video extension. Allowed: {allowed}")

    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=500, detail=f"Model file not found: {MODEL_PATH}")

    request_id = uuid.uuid4().hex
    temporary_directory = Path(tempfile.mkdtemp(prefix="faceforensics-"))
    upload_path = temporary_directory / f"upload{suffix}"
    result_path = API_OUTPUT_DIR / f"{request_id}.json"

    try:
        written_bytes = 0
        with upload_path.open("wb") as upload_destination:
            while chunk := await file.read(1024 * 1024):
                written_bytes += len(chunk)
                upload_destination.write(chunk)

        if written_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

        API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "video.classification.detect_from_video",
            "--video_path",
            str(upload_path),
            "--model_path",
            str(MODEL_PATH),
            "--output_path",
            str(API_OUTPUT_DIR),
            "--result_path",
            str(result_path),
            "--frame_interval",
            "5",
        ]

        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                timeout=INFERENCE_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as error:
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS} seconds.",
            ) from error
        except subprocess.CalledProcessError as error:
            raise HTTPException(status_code=500, detail=_process_error_detail(error)) from error

        if not result_path.is_file():
            raise HTTPException(status_code=500, detail="Inference completed but did not create a JSON result.")

        try:
            with result_path.open("r", encoding="utf-8") as result_file:
                result = json.load(result_file)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail="Inference result JSON could not be parsed.") from error

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Inference result JSON must be an object.")

        result["status"] = "success"
        return result
    finally:
        await file.close()
        shutil.rmtree(temporary_directory, ignore_errors=True)
