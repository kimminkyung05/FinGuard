# 영상 딥페이크 위험도 분석 파이프라인

이 프로젝트는 Windows 시스템 Python 3.13에서 실행합니다. Miniconda, dlib,
CMake, 별도 가상환경을 사용하지 않습니다.

## 설치

```powershell
python --version
python -m pip install -r requirements.txt
```

`Python 3.13.x`가 출력되는지 확인하세요. Jupyter에서는
`FaceForensics (Python 3.13)` 커널을 선택합니다.

## 실행

```powershell
python -m video.classification.detect_from_video `
  --video_path video\sample1.mp4 `
  --model_path video\face_detection\xception\all_c23.p `
  --output_path outputs\xception_inference
```

OpenCV의 기본 Haar Cascade가 가장 큰 얼굴을 검출하고, margin을 적용해 crop한 뒤
기존 Xception checkpoint로 프레임별 fake score를 계산합니다. 결과 폴더에는 annotated
AVI와 금융권 연동용 JSON이 생성됩니다. AVI가 필요 없으면
`--no_save_annotated_video`를 추가하세요.

## FastAPI 영상 분석 API

서버를 실행합니다.

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

브라우저에서 Swagger UI를 엽니다.

```text
http://127.0.0.1:8000/docs
```

`POST /video/analyze`를 열고 **Try it out**을 누른 뒤, `file` 항목에 영상을
선택하여 실행합니다. `.mp4`, `.avi`, `.mov`, `.mkv` 파일을 지원합니다.

업로드된 영상은 임시 경로에 저장된 뒤 기존 Xception 영상 추론 CLI로 처리됩니다.
성공하면 생성된 JSON 결과를 그대로 반환하며, 최상위 `status`는 `success`입니다.

응답에는 다음 점수가 포함됩니다.

```json
{
  "status": "success",
  "scores": {
    "video_fake_score": 0.9325,
    "confidence_score": 0.7810
  }
}
```

간단한 상태 확인은 다음 주소를 사용합니다.

```text
GET http://127.0.0.1:8000/health
```

## 주의 사항

`all_c23.p`는 구형 PyTorch에서 저장된 신뢰된 로컬 모델 객체입니다. 최신 PyTorch에서
호환 로딩을 위해 `weights_only=False`와 기존 `network.*` 모듈 경로 alias를 사용합니다.
출처를 신뢰할 수 없는 checkpoint에는 사용하면 안 됩니다.
