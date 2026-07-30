# DeepFake 영상 분석 API

업로드한 영상에서 얼굴 구간을 탐지하고 Xception 모델로 딥페이크 위험도를 분석하는 FastAPI 서버입니다.

## 준비

PowerShell에서 이 폴더를 작업 위치로 설정합니다.

```powershell
cd C:\Users\user\Desktop\minkyung\FaceForensics\DeepFake
python -m pip install -r requirements.txt
```

모델 파일이 아래 경로에 있어야 합니다.

```text
video\face_detection\xception\all_c23.p
```

## API 서버 실행

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

서버가 실행 중인 터미널은 종료하지 않은 채 브라우저에서 Swagger UI를 엽니다.

```text
http://127.0.0.1:8000/docs
```

## Swagger에서 영상 분석 테스트

1. Swagger UI에서 `POST /video/analyze`를 펼칩니다.
2. **Try it out**을 누릅니다.
3. `file` 항목의 **Choose File**에서 분석할 영상 파일을 선택합니다. 지원 형식은 `.mp4`, `.avi`, `.mov`, `.mkv`입니다.
4. **Execute**를 누릅니다.
5. 응답 코드가 `200`이면 분석 결과 JSON을 확인합니다. `scores.video_fake_score`는 딥페이크 점수, `scores.confidence_score`는 분석 신뢰도입니다.

먼저 서버 상태만 확인하려면 Swagger의 `GET /health`에서 **Try it out** → **Execute**를 누르거나 아래 주소를 엽니다.

```text
http://127.0.0.1:8000/health
```

정상 응답은 다음과 같습니다.

```json
{"status": "ok"}
```

## 오류 확인

- `500 Model file not found`: 위 모델 파일 경로와 파일명을 확인합니다.
- `400 Unsupported video extension`: 지원되는 영상 확장자로 업로드합니다.
- `504 Inference timed out`: 영상 길이 또는 실행 환경을 확인한 뒤 다시 시도합니다.
