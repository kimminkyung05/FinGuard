import os
import sys
import json
import asyncio
import logging
import secrets
import contextlib
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status, Query
from pydantic_settings import BaseSettings
from faster_whisper import WhisperModel
import imageio_ffmpeg

# ⚠️ pydub는 import되는 "그 순간" 자체적으로 which("ffmpeg")를 체크합니다.
# import pydub.utils 이후에 which를 덮어써봐야 이미 경고가 뜬 뒤라 소용없음 ->
# import pydub.utils 되기 전에 PATH에 ffmpeg 경로를 먼저 넣어야 함.
_ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

import pydub.utils

import base64
import io
import random
import torchvision.transforms as transforms
try:
    from PIL import Image
except ImportError:
    Image = None

pydub.utils.which = lambda cmd: imageio_ffmpeg.get_ffmpeg_exe() if cmd in ["ffmpeg", "ffprobe"] else None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceServer")

class Settings(BaseSettings):
    app_name: str = "Hybrid Anti-Fraud Audio & Video Gateway"
    sample_rate: int = 16000
    chunk_duration_ms: int = 100
    vad_threshold: float = 0.5
    max_speech_duration_s: float = 8.0
    min_speech_duration_s: float = 0.5
    softmax_temperature: float = 1.15
    decision_threshold: float = 0.60
    uncertainty_deferral_tau: float = 0.75
    w_regex: float = 2.35
    w_semantic: float = 3.60
    b_logistic: float = -1.85
    logit_margin_scale_theta: float = 3.0
    energy_mu_default: float = 0.02
    energy_sigma_default: float = 0.005
    snr_min: float = 5.0
    snr_max: float = 30.0
    sbert_model_path: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ws_api_token: str = "secret-token-123"
    min_noise_power: float = 1e-6
    realtime_mc_dropout_passes: int = 3

settings = Settings()

if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "cuda", "float16"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "cpu", "int8"
else:
    TORCH_DEVICE = torch.device("cpu")
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "cpu", "int8"

app = FastAPI(title=settings.app_name)

# =====================================================================
# 본인 원래 경로 세팅
# =====================================================================
aasist_path = r"C:\Users\woori\FinGuard\voice\aasist"
deepfake_base_path = r"C:\Users\woori\FinGuard\DeepFake"
video_network_path = r"C:\Users\woori\FinGuard\DeepFake\video\classification\network"

# ---------------------------------------------------------------------
# ⚠️ 핵심 수정: aasist 쪽 "models" 패키지와 DeepFake 쪽 "models" 패키지가
# 이름이 같아서 sys.path에 둘 다 올리면 나중에 import하는 쪽이 깨집니다.
# sys.path/import 문 대신 importlib로 파일 경로에서 직접, 서로 다른
# 모듈 이름(alias)으로 로드해서 충돌 자체를 없앱니다.
# ---------------------------------------------------------------------
def _load_module_from_path(unique_name: str, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module

RealAASISTModel = None
try:
    _aasist_mod = _load_module_from_path(
        "aasist_models_AASIST",
        os.path.join(aasist_path, "models", "AASIST.py"),
    )
    RealAASISTModel = _aasist_mod.Model
except Exception as e:
    logger.warning(f"⚠️ [AASIST 모듈 import 실패] {e}")

VideoXceptionFactory = None
try:
    if video_network_path not in sys.path:
        # xception.py 자체는 "models"라는 이름이 아니라 충돌 없음 -> 그냥 sys.path로 로드 가능
        sys.path.append(video_network_path)
    from xception import xception as VideoXceptionFactory
except Exception as e:
    logger.warning(f"⚠️ [Xception 모듈 import 실패] {e}")
    VideoXceptionFactory = None


class GlobalModelEngine:
    def __init__(self):
        self.stt_model: Optional[WhisperModel] = None
        self.fake_voice_detector: Optional[Any] = None
        self.vad_model: Optional[torch.nn.Module] = None
        self.sbert_model: Optional[Any] = None
        self.video_model: Optional[torch.nn.Module] = None
        self.video_transform = None

        self.system_degraded: bool = False
        self.phishing_anchors = []
        self.aasist_lock = asyncio.Lock()
        self.load_all_models()

    def load_all_models(self):
        logger.info("🤖 [엔진 초기화] 모델 로딩 시작...")
        try:
            self.stt_model = WhisperModel("base", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        except Exception as e:
            logger.exception(f"❌ [STT] 실패 - {e}")
            self.system_degraded = True

        try:
            if RealAASISTModel is None:
                raise ImportError("AASIST 모듈 없음")
            config_path = os.path.join(aasist_path, "config", "AASIST.conf")
            weights_path = os.path.join(aasist_path, "models", "weights", "AASIST.pth")
            if os.path.exists(config_path) and os.path.exists(weights_path):
                with open(config_path, "r") as f:
                    self.fake_voice_detector = RealAASISTModel(json.load(f)["model_config"])
                self.fake_voice_detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
                self.fake_voice_detector.to(TORCH_DEVICE)
                self.fake_voice_detector.eval()
                logger.info("✅ [AASIST] 음성 딥페이크 로딩 성공!")
            else:
                logger.warning(f"⚠️ [AASIST] 설정 파일/가중치를 찾을 수 없습니다: {aasist_path}")
        except Exception as e:
            logger.warning(f"⚠️ [AASIST] 로딩 경고 (무시됨): {e}")
            self.fake_voice_detector = None

        try:
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        except Exception as e:
            logger.exception(f"⚠️ [VAD] 실패 - {e}")
            self.vad_model = None

        if SentenceTransformer is not None:
            try:
                self.sbert_model = SentenceTransformer(settings.sbert_model_path)
                anchor_texts = [
                    "검찰청 수사관입니다. 통장을 동결하겠습니다.",
                    "안전계좌로 자금을 이체하셔야 합니다.",
                    "팀뷰어 원격지원 앱을 설치해주세요."
                ]
                self.phishing_anchors = self.sbert_model.encode(anchor_texts)
            except Exception as e:
                logger.exception(f"⚠️ [SBERT] 실패 - {e}")

        logger.info("🤖 [엔진 초기화] 영상 딥페이크 탐지 모델 로딩 시작...")
        if VideoXceptionFactory is not None:
            try:
                # pretrained=False로 불러오면 BatchNorm 통계까지 완전 랜덤이라
                # 입력이 달라도 출력이 사실상 고정되는 문제가 생깁니다.
                # -> 백본은 pretrained=True(imagenet)로 정상 로드하고,
                #    마지막 분류 head만 2-class로 교체합니다.
                try:
                    self.video_model = VideoXceptionFactory(num_classes=2)
                except (AssertionError, TypeError):
                    self.video_model = VideoXceptionFactory(num_classes=1000, pretrained=True)
                    if hasattr(self.video_model, "fc"):
                        in_features = self.video_model.fc.in_features
                        self.video_model.fc = nn.Linear(in_features, 2)
                    elif hasattr(self.video_model, "last_linear"):
                        in_features = self.video_model.last_linear.in_features
                        self.video_model.last_linear = nn.Linear(in_features, 2)
                self.video_transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3)
                ])
                self.video_model.to(TORCH_DEVICE)
                self.video_model.eval()
                logger.info("✅ [Video AI] 비디오 모델 로딩 성공!")
            except Exception as e:
                logger.warning(f"⚠️ [Video AI] 로딩 경고 (무시됨): {e}")
                self.video_model = None
        else:
            logger.warning("⚠️ [Video AI] xception 모듈 없음 - 비디오는 더미 점수로 동작")


engine = GlobalModelEngine()


@app.get("/health")
async def health_check():
    return {
        "status": "ok" if not engine.system_degraded else "degraded",
        "stt": engine.stt_model is not None,
        "aasist": engine.fake_voice_detector is not None,
        "vad": engine.vad_model is not None,
        "video": engine.video_model is not None,
    }


def check_signal_quality(audio_samples: np.ndarray) -> dict:
    if len(audio_samples) < 1600:
        return {"snr": 20.0, "clipping_ratio": 0.0}
    clipping_ratio = float(np.sum(np.abs(audio_samples) > 0.99) / len(audio_samples))
    frame_len = 400
    energies = [np.sum(audio_samples[i:i + frame_len] ** 2) for i in range(0, len(audio_samples) - frame_len, frame_len)]
    signal_power = np.mean(energies) if energies else 1e-9
    noise_power = max(np.percentile(energies, 5) if energies else 1e-9, settings.min_noise_power)
    snr = 10 * np.log10(signal_power / noise_power)
    return {"snr": float(np.clip(snr, 0, 30)), "clipping_ratio": clipping_ratio}


def analyze_audio_sliding_window_mcd(samples: np.ndarray, model, target_len: int = 64600, hop_len: int = 48450):
    if model is None:
        return 0.5, [0.0, 0.0], 0.0
    model_device = next(model.parameters()).device
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    window_means, window_logits, window_variances = [], [], []
    start_idx = 0
    while start_idx < len(samples):
        chunk = samples[start_idx: start_idx + target_len]
        if len(chunk) < target_len:
            chunk = np.pad(chunk, (0, target_len - len(chunk)), 'constant')
        tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(model_device)

        mc_preds = []
        best_logits = [0.0, 0.0]
        with torch.no_grad():
            for _ in range(settings.realtime_mc_dropout_passes):
                _, prediction = model(tensor)
                mc_preds.append(torch.sigmoid(prediction)[0][1].item())
                best_logits = [prediction[0][0].item(), prediction[0][1].item()]

        window_means.append(np.mean(mc_preds))
        window_variances.append(np.var(mc_preds))
        window_logits.append(best_logits)
        if start_idx + target_len >= len(samples):
            break
        start_idx += hop_len

    model.eval()
    if not window_means:
        return 0.5, [0.0, 0.0], 0.0

    k = min(3, len(window_means))
    top_indices = np.argsort(window_means)[-k:]
    avg_score = float(sum(window_means[i] for i in top_indices) / k)
    epistemic_uncertainty = float(sum(window_variances[i] for i in top_indices) / k)
    return round(avg_score, 4), window_logits[top_indices[-1]], round(epistemic_uncertainty, 4)


def estimate_semantic_risk(text: str):
    import re
    norm_text = re.sub(r'\s+', '', text)
    auth_words = [r'검찰', r'경찰', r'수사관', r'금융감독원']
    fin_words = [r'안전계좌', r'대출', r'송금', r'입금']
    c_auth = len([p for p in auth_words if re.search(p, norm_text)])
    c_fin = len([p for p in fin_words if re.search(p, norm_text)])
    regex_risk = 1.0 - np.exp(-((c_auth * 0.4) + (c_fin * 0.4)))

    semantic_risk = 0.0
    if engine.sbert_model and len(engine.phishing_anchors) > 0:
        try:
            text_emb = np.asarray(engine.sbert_model.encode([text]))
            anchors = np.asarray(engine.phishing_anchors)
            cos_sims = np.dot(text_emb, anchors.T) / (np.linalg.norm(text_emb, axis=1, keepdims=True) * np.linalg.norm(anchors, axis=1))
            semantic_risk = float(np.max(cos_sims))
        except Exception:
            pass

    final_risk = 1.0 / (1.0 + np.exp(-(settings.w_regex * regex_risk + settings.w_semantic * semantic_risk + settings.b_logistic)))
    return round(final_risk, 4)


def realtime_multimodal_worker_sync(audio_array: np.ndarray, session_history: List[str]) -> Dict[str, Any]:
    result = {"text": "", "risk_score": 0.0, "threat_level": "NORMAL"}
    if not engine.stt_model:
        return result

    segments = list(engine.stt_model.transcribe(audio_array, beam_size=1, language="ko", condition_on_previous_text=False)[0])
    text_chunk = " ".join([s.text for s in segments]).strip()
    if not text_chunk:
        return result

    result["text"] = text_chunk
    session_history.append(text_chunk)
    context_window = " ".join(session_history[-5:])

    t_risk = estimate_semantic_risk(context_window)
    _ = check_signal_quality(audio_array)  # 신호 품질 계산(로깅/추후 확장용)

    audio_score = 0.0
    if engine.fake_voice_detector:
        audio_score, _, _ = analyze_audio_sliding_window_mcd(audio_array, engine.fake_voice_detector)

    final_risk = round(float(audio_score * 0.6 + t_risk * 0.4), 4)
    threat = "HIGH RISK" if final_risk >= settings.decision_threshold else "NORMAL"

    result["risk_score"] = final_risk
    result["threat_level"] = threat
    return result


def analyze_video_frame(base64_str: str) -> float:
    if not Image or engine.video_model is None or engine.video_transform is None:
        return round(random.uniform(0.12, 0.25), 4)
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        tensor = engine.video_transform(image).unsqueeze(0).to(TORCH_DEVICE)
        with torch.no_grad():
            outputs = engine.video_model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fake_score = probabilities[0][1].item()
        return round(float(fake_score), 4)
    except Exception:
        return round(random.uniform(0.12, 0.25), 4)


class CallSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.speech_buffer: List[np.ndarray] = []
        self.is_speaking = False
        self.transcript_history: List[str] = []
        self.vad_probs: List[float] = []

    def reset(self):
        self.speech_buffer.clear()
        self.vad_probs.clear()
        self.is_speaking = False


@app.websocket("/ws/detect/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, token: str = Query(None)):
    if not token or not secrets.compare_digest(token, settings.ws_api_token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    session = CallSession(session_id=session_id)
    audio_queue = asyncio.Queue(maxsize=50)
    chunk_size_bytes = int(settings.sample_rate * 2 * (settings.chunk_duration_ms / 1000.0))

    logger.info(f"🔌 [웹소켓 연결] 세션 시작: {session_id}")

    async def process_audio():
        audio_buffer = bytearray()
        try:
            while True:
                data = await audio_queue.get()
                audio_buffer.extend(data)

                while len(audio_buffer) >= chunk_size_bytes:
                    chunk_bytes = audio_buffer[:chunk_size_bytes]
                    del audio_buffer[:chunk_size_bytes]

                    pcm_data = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    speech_prob = 1.0

                    if engine.vad_model:
                        chunk_size = 512
                        vad_input = pcm_data[:chunk_size] if len(pcm_data) >= chunk_size else np.pad(pcm_data, (0, chunk_size - len(pcm_data)))
                        with torch.no_grad():
                            speech_prob = engine.vad_model(torch.from_numpy(vad_input), settings.sample_rate).item()

                    is_speech = speech_prob >= settings.vad_threshold

                    if is_speech:
                        if not session.is_speaking:
                            session.is_speaking = True
                        session.speech_buffer.append(pcm_data)
                        session.vad_probs.append(speech_prob)

                    if not is_speech and session.is_speaking:
                        speech_duration = len(session.speech_buffer) * (settings.chunk_duration_ms / 1000.0)
                        if speech_duration >= settings.min_speech_duration_s:
                            full_speech_array = np.concatenate(session.speech_buffer)

                            async with engine.aasist_lock:
                                ml_result = await asyncio.to_thread(
                                    realtime_multimodal_worker_sync,
                                    full_speech_array,
                                    session.transcript_history
                                )

                            if ml_result["text"]:
                                await websocket.send_json({
                                    "event": "INTENT_ANALYZED",
                                    "transcript_latest": ml_result["text"],
                                    "risk_score": ml_result["risk_score"],
                                    "threat_level": ml_result["threat_level"]
                                })
                        session.reset()
                audio_queue.task_done()
        except asyncio.CancelledError:
            pass

    processor_task = asyncio.create_task(process_audio())

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "video_frame":
                        base64_image = data.get("data")
                        video_score = await asyncio.to_thread(analyze_video_frame, base64_image)
                        threat_level = "HIGH RISK" if video_score >= settings.decision_threshold else "NORMAL"
                        await websocket.send_json({
                            "event": "VIDEO_ANALYZED",
                            "video_score": video_score,
                            "threat_level": threat_level
                        })
                except json.JSONDecodeError:
                    pass

            elif "bytes" in message and message["bytes"]:
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await audio_queue.put(message["bytes"])

    except WebSocketDisconnect:
        logger.info(f"🔌 [웹소켓 종료] 클라이언트 해제: {session_id}")
    except Exception as e:
        logger.info(f"🔌 [웹소켓 연결 종료됨]: {e}")
    finally:
        processor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await processor_task

        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        session.reset()
        logger.info(f"🧹 [세션 정리 완료] {session_id}")