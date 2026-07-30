# 영상 분류 실행

이 추론 경로는 Windows 시스템 Python 3.13을 직접 사용합니다. Miniconda, dlib,
CMake, 별도 가상환경은 필요하지 않습니다.

```powershell
python -m pip install -r requirements.txt
python -m video.classification.detect_from_video `
  --video_path video\sample1.mp4 `
  --model_path video\face_detection\xception\all_c23.p `
  --output_path outputs\xception_inference
```

`detect_from_video.py`는 OpenCV Haar Cascade로 가장 큰 얼굴을 검출합니다. 검출한
bounding box에 margin을 적용해 crop한 뒤 기존 Xception checkpoint를 실행하고,
annotated AVI 및 JSON 결과를 출력합니다. AVI가 불필요하면
`--no_save_annotated_video`를 추가하세요.
