# FaceForensics

FaceForensics는 영상 딥페이크와 음성 스푸핑/보이스피싱 징후를 분석하는 실험용 안티프로드 프로젝트입니다.

- FaceForensics 데이터 기반 Xception 모델을 이용한 영상 딥페이크 분석
- AASIST, Whisper, VAD, 텍스트 위험 신호를 이용한 음성 분석



## 🎙️ 주요 음성 분석 기능 (상세)

- **QC 필터링 및 신호 품질 검사**: 오디오의 SNR(신호 대 잡음비)을 측정하고, 신호의 클리핑(Clipping) 비율을 탐지하여 분석 신뢰도 가중치에 동적으로 반영합니다.
- **비언어 특징 분석 (Prosody)**: 발화에서 Jitter, Shimmer, HNR, F0(기본 주파수) 분산, 발화 속도, MFCC 분산 등 9가지의 파라링구이스틱(Paralinguistic) 비언어 특징을 추출하여 모델 융합에 활용합니다.
- **하이브리드 의미론적 위험 (Semantic Risk)**: SBERT 기반의 피싱 앵커 텍스트 유사도 검사와 강력한 정규식을 결합하여, 대화 내 금전 요구 및 사칭 키워드 의도를 정밀하게 탐지합니다.
- **문맥 유지형 동적 STT (Whisper large-v3)**: 실시간 스트림에서 화자의 이전 대화 이력을 프롬프트로 지속 재사용하여, 끊김 없이 문맥이 유지되는 고정밀 음성 인식을 수행합니다.
- **신뢰도 기반 융합 (Gating Fusion)**: 음성 스푸핑 점수, 텍스트 위험도, 비언어 점수의 충돌(Jensen-Shannon 발산)과 몬테카를로 드롭아웃(MCD) 분산을 계산하여 시스템의 불확실성을 측정하고 가장 신뢰할 수 있는 데이터에 가중치를 부여합니다.

## 저장소 구성

| 경로 | 설명 |
| --- | --- |
| `DeepFake/` | FastAPI 기반 영상 업로드 API와 Xception 영상 추론 파이프라인 |
| `voice/` | 음성 분석 API, 실시간 WebSocket 워커, 보조 스크립트 |
| `voice/aasist/` | AASIST 학습, 평가, 사전학습 가중치 코드 |
| `index.html` | 카메라와 마이크 입력을 위한 로컬 브라우저 프로토타입 |

## 사전 요구 사항

- Windows PowerShell
- Python 3.11 이상
- Git
- 브라우저 프로토타입 사용 시 웹캠과 마이크
- FFMPEG (음성 전처리 및 pydub 작동을 위해 시스템 `PATH`에 추가 필요)

## 가상환경 설정

패키지를 설치하거나 서버를 실행하기 전에 프로젝트 전용 가상환경을 생성하고 활성화합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

현재 PowerShell 세션에서 실행 정책 때문에 활성화 스크립트가 차단되면 다음을 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

가상환경이 활성화되면 프롬프트 앞에 `(.venv)`가 표시됩니다. 새 터미널을 열었을 때도 작업 전에 같은 가상환경을 다시 활성화해야 합니다.

## 의존성 설치

### 영상 API

```powershell
python -m pip install -r DeepFake\requirements.txt
```

Xception 추론에 필요한 FastAPI, OpenCV, PyTorch, TorchVision 등을 설치합니다.

### 음성 API

```powershell
python -m pip install -r voice\aasist\requirements.txt
python -m pip install fastapi "uvicorn[standard]" python-multipart pydantic-settings faster-whisper pydub imageio-ffmpeg scipy praat-parselmouth transformers sentence-transformers joblib
```

음성 분석 스택은 최초 실행 시 일부 모델 파일을 내려받을 수 있습니다. 처음 실행할 때는 인터넷 연결과 충분한 디스크 공간이 필요합니다.

## 영상 딥페이크 API

API는 로컬 체크포인트 `DeepFake/video/face_detection/xception/all_c23.p`를 사용합니다.

저장소 최상위에서 서버를 실행합니다.

```powershell
python -m uvicorn DeepFake.app.main:app --host 127.0.0.1 --port 8000
```

대화형 API 문서:

```text
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
```

상태 확인:

```powershell
Invoke-RestMethod [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
```

영상 분석 엔드포인트는 `POST /video/analyze`이며 `.mp4`, `.avi`, `.mov`, `.mkv` 업로드를 지원합니다.

### 영상 결과 해석

응답에는 다음 주요 필드가 포함됩니다.

- `scores.video_fake_score`: 유효 얼굴 프레임에서 집계한 Xception 가짜 점수
- `scores.confidence_score`: 분석 품질과 일관성 평가 점수
- `risk.decision`: `APPROVE`, `RETRY`, `BLOCK`
- `frame_analysis.statistics`: 평균, 최솟값/최댓값, 상위 구간 평균, 고위험 프레임 비율

모델의 원시 점수는 보정된 확률이 아닙니다. 운영 임계값을 적용하기 전, 실제 사용 환경을 대표하는 라벨 데이터로 점수 보정을 수행해야 합니다.

## 음성 API

음성 WebSocket은 토큰이 필요합니다. 서버 실행 전 활성화된 터미널에서 토큰을 설정합니다.

```powershell
$env:FRAUD_WS_API_TOKEN = "replace-with-a-long-random-token"
python -m uvicorn voice.server:app --host 127.0.0.1 --port 8001
```

상태 확인:

```powershell
Invoke-RestMethod [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health)
```

엔드포인트는 시스템 과부하(System Degraded) 상태 및 탑재된 모델별 로드 상태를 반환합니다.

배치 분석 엔드포인트는 `POST /analyze/pipeline`이며, 실시간 엔드포인트는 `WS /ws/detect/{session_id}`입니다.

### 음성 결과 해석 및 XAI 리포트

하이브리드 융합 엔진에 의해 도출된 최종 결과는 다음 4단계 위협 수준(Threat Level) 중 하나로 반환됩니다.
- `NORMAL`: 정상 범주
- `MODERATE RISK`: 의심스러운 패턴 감지
- `HIGH RISK`: 명백한 보이스피싱 또는 딥페이크 징후
- `MANUAL REVIEW (HIGH UNCERTAINTY)`: 입력 신호 간 충돌(JSD) 및 불확실성이 임계치(Tau)를 초과하여 상담원 개입이 필요한 상태

또한 융합 결과에는 모달리티 간 충돌 수치(`tri_modal_conflict_jsd`), 모델 불확실성(`system_uncertainty`), 각 분석기(음향, 텍스트, 운율)에 부여된 동적 신뢰도 가중치(`gating_weights`)가 포함됩니다.

## 브라우저 프로토타입

`index.html`은 카메라와 마이크 권한을 요청하는 로컬 프로토타입입니다. 현재 개발용 WebSocket 주소와 토큰이 코드에 고정되어 있으므로, 사용 전 음성 서버 주소와 `FRAUD_WS_API_TOKEN`에 맞게 변경해야 합니다. 배포 환경의 클라이언트 코드에는 토큰을 포함하지 마세요.

## 테스트

저장소 최상위에서 영상 의사결정 레이어 테스트를 실행합니다.

```powershell
python -m unittest DeepFake.tests.test_video_service -v
```

이 테스트는 점수 집계, 신뢰도, 의사결정 정책을 검증합니다. 모델 정확도나 영상/음성 데이터셋 기반 성능 평가는 별도로 수행해야 합니다.

## 보안 및 운영 주의 사항

- 모델 파일은 신뢰할 수 있는 출처의 파일만 사용하세요. 기존 Xception 체크포인트는 직렬화된 PyTorch 모델 객체로 로드됩니다.
- API 토큰, 업로드 영상, 생성 영상, 개인 정보를 포함할 수 있는 분석 결과는 Git에 커밋하지 마세요.
- C23 Xception 체크포인트는 FaceForensics 형식의 압축 영상으로 학습되었습니다. 웹캠, 화면 녹화, 재인코딩 영상은 학습 분포와 달라 과신 점수가 나올 수 있습니다.
- `RETRY` 결과는 재촬영 또는 사람의 검토 흐름으로 연결하세요. 원시 모델 점수를 보정 확률로 해석하면 안 됩니다.
- **음성 강제 오버라이드 규칙**: 음성 분석 중 텍스트 엔진이 '돈', '송금', '계좌' 등 명백한 금전 요구 키워드를 감지하면, 시스템은 다른 신뢰도 지표를 무시하고 텍스트 위험도를 강제로 최대치(0.99)로 재정의합니다.

## 라이선스 및 출처

저장소에는 서드파티 FaceForensics 및 AASIST 자료가 포함되어 있습니다. 재배포 또는 상업적 사용 전 각 프로젝트에 포함된 라이선스와 고지 파일을 확인하세요.
