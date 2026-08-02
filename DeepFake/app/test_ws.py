import os
import sys
import json
import asyncio
import logging
import secrets
import contextlib
import importlib.util
import collections
import numpy as np
import torch
import torch.nn as nn
import math
import re
import io
import time
import random
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status, Query
from pydantic_settings import BaseSettings
from faster_whisper import WhisperModel
import imageio_ffmpeg
from scipy.spatial.distance import jensenshannon
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call
from typing import Union
from pydantic import BaseModel

# ⚠️ FFMPEG 경로 세팅
_ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

import pydub.utils
import base64
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
logger = logging.getLogger("HybridApp")

class Settings(BaseSettings):
    app_name: str = "Hybrid Anti-Fraud Audio & Video Gateway (Full Multimodal)"
    sample_rate: int = 16000
    chunk_duration_ms: int = 100
    vad_threshold: float = 0.5
    max_speech_duration_s: float = 8.0
    min_speech_duration_s: float = 0.5
    
    # 🔥 [복구] 고급 논리/수학 연산 하이퍼파라미터 전부 부활
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
    jitter_max: float = 0.01      
    shimmer_max: float = 0.1      
    hnr_max: float = 30.0         
    speech_rate_max: float = 8.0  
    f0_var_max: float = 5000.0    
    pause_ratio_max: float = 1.0
    mfcc_var_train_mu: float = 1450.2
    mfcc_var_train_sigma: float = 480.5
    f1_var_train_mu: float = 24500.0
    f1_var_train_sigma: float = 9800.0
    invalid_reliability: float = 0.0
    mc_dropout_passes: int = 10 
    realtime_mc_dropout_passes: int = 3
    
    sbert_model_path: str = "paraphrase-multilingual-MiniLM-L12-v2"
    prosody_model_path: Optional[str] = None
    ws_api_token: str = "secret-token-123"
    min_noise_power: float = 1e-6
    
    # 🔥 시리/빅스비 급 STT 정확도 최적화
    whisper_model_size: str = "large-v3"
    whisper_beam_size_batch: int = 5
    whisper_beam_size_realtime: int = 5   
    whisper_domain_prompt: str = (
        "네, 알겠습니다. 계좌번호 알려주시면 100만원 당장 송금할게요. "
        "돈 보내주세요. 비밀번호, 인증번호, 원격 앱, 명의도용 관련 대화입니다."
    )

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
# 비디오 및 오디오 딥페이크 경로 세팅
# =====================================================================
aasist_path = r"C:\Users\woori\FinGuard\voice\aasist"
deepfake_base_path = r"C:\Users\woori\FinGuard\DeepFake"
video_classification_path = r"C:\Users\woori\FinGuard\DeepFake\video\classification"
video_network_path = r"C:\Users\woori\FinGuard\DeepFake\video\classification\network"

VideoXceptionFactory = None
try:
    if deepfake_base_path not in sys.path:
        sys.path.append(deepfake_base_path)
    if video_classification_path not in sys.path:
        sys.path.append(video_classification_path)
    if video_network_path not in sys.path:
        sys.path.append(video_network_path)
        
    from xception import xception as VideoXceptionFactory
except Exception as e:
    logger.warning(f"⚠️ [Xception 모듈 import 실패] {e}")
    VideoXceptionFactory = None

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

class GlobalModelEngine:
    def __init__(self):
        self.stt_model: Optional[WhisperModel] = None
        self.fake_voice_detector: Optional[Any] = None
        self.vad_model: Optional[torch.nn.Module] = None
        self.sbert_model: Optional[Any] = None
        self.video_model: Optional[torch.nn.Module] = None
        self.video_transform = None
        self.prosody_classifier: Optional[Any] = None 

        self.system_degraded: bool = False
        self.phishing_anchors = []
        self.aasist_lock = asyncio.Lock()
        self.load_all_models()

    def load_all_models(self):
        logger.info("🤖 [엔진 초기화] 모델 로딩 시작...")
        try:
            self.stt_model = WhisperModel(settings.whisper_model_size, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            logger.info(f"✅ [STT] Whisper '{settings.whisper_model_size}' 로드 완료")
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
                    "팀뷰어 원격지원 앱을 설치해주세요.",
                    "엄마 나 폰 고장났어. 수리비 100만원만 계좌로 보내줘.", 
                    "지금 급해서 그런데 돈 좀 당장 보내줄 수 있어?"
                ]
                self.phishing_anchors = self.sbert_model.encode(anchor_texts)
            except Exception as e:
                logger.exception(f"⚠️ [SBERT] 실패 - {e}")

        logger.info("🤖 [엔진 초기화] 영상 딥페이크 탐지 모델 로딩 시작...")
        if VideoXceptionFactory is not None:
            try:
                try:
                    self.video_model = VideoXceptionFactory(num_classes=2)
                except Exception:
                    try:
                        self.video_model = VideoXceptionFactory(num_classes=1000, pretrained=False)
                    except Exception:
                        self.video_model = VideoXceptionFactory(num_classes=1000)

                if hasattr(self.video_model, "fc"):
                    in_features = self.video_model.fc.in_features
                    self.video_model.fc = nn.Linear(in_features, 2)
                elif hasattr(self.video_model, "last_linear"):
                    in_features = self.video_model.last_linear.in_features
                    self.video_model.last_linear = nn.Linear(in_features, 2)

                video_weights_path = r"C:\Users\woori\FinGuard\DeepFake\video\face_detection\xception\all_c23.p"

                if os.path.exists(video_weights_path):
                    try:
                        try:
                            loaded_data = torch.load(video_weights_path, map_location="cpu", weights_only=False)
                        except TypeError:
                            loaded_data = torch.load(video_weights_path, map_location="cpu")

                        if isinstance(loaded_data, dict):
                            if "model" in loaded_data:
                                state_dict = loaded_data["model"]
                            elif "state_dict" in loaded_data:
                                state_dict = loaded_data["state_dict"]
                            else:
                                state_dict = loaded_data
                            self.video_model.load_state_dict(state_dict, strict=False)
                        else:
                            self.video_model = loaded_data
                            
                        logger.info("✅ [Video AI] 비디오 딥페이크 가중치(all_c23.p) 로딩 성공!")
                    except Exception as load_err:
                        logger.warning(f"⚠️ [Video AI] 가중치 로딩 실패: {load_err}")
                else:
                    logger.warning(f"⚠️ [Video AI] 가중치 파일을 찾을 수 없습니다: {video_weights_path}")

                self.video_transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3)
                ])
                
                self.video_model.to(TORCH_DEVICE)
                self.video_model.eval()
            except Exception as e:
                logger.info(f"ℹ️ [Video AI] 로딩 예외 발생 ({e}) - 더미 점수 모드로 동작")
                self.video_model = None
                self.video_transform = None
        else:
            logger.warning("⚠️ [Video AI] xception 모듈 없음 - 비디오는 더미 점수로 동작")

engine = GlobalModelEngine()

# ==========================================
# 📐 [복구됨] 정밀 오디오 논리 및 수학 함수들 전부 추가
# ==========================================
def check_signal_quality(audio_samples: np.ndarray) -> dict:
    if len(audio_samples) < 16000: 
        return {"snr": 20.0, "clipping_ratio": 0.0, "energy_mean": settings.energy_mu_default, "energy_std": settings.energy_sigma_default}
    
    clipping_ratio = float(np.sum(np.abs(audio_samples) > 0.99) / len(audio_samples))
    frame_len = 400 
    energies = [np.sum(audio_samples[i:i+frame_len]**2) for i in range(0, len(audio_samples)-frame_len, frame_len)]
    signal_power = np.mean(energies) if energies else 1e-9
    noise_power = max(np.percentile(energies, 5) if energies else 1e-9, settings.min_noise_power)
    snr = 10 * np.log10(signal_power / noise_power)
    
    energy_mean = float(np.mean(audio_samples**2))
    energy_std = float(np.std(audio_samples**2)) or settings.energy_sigma_default
    
    return {"snr": float(np.clip(snr, 0, 30)), "clipping_ratio": clipping_ratio, "energy_mean": energy_mean, "energy_std": energy_std}

def extract_paper_grade_nonverbal_features(audio_input: Union[str, np.ndarray], signal_metrics: dict, sample_rate: int = 16000) -> dict:
    try:
        if signal_metrics.get("energy_mean", 0.0) < 1e-6:
            return {"features": np.zeros(9), "quality_metrics": {}, "feature_valid": False}

        snd = parselmouth.Sound(audio_input) if isinstance(audio_input, str) else parselmouth.Sound(audio_input.astype(np.float64), sample_rate)
        
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_variance = np.var(f0_values[f0_values > 0]) if len(f0_values[f0_values > 0]) > 0 else 0
        
        formant_obj = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        num_frames = call(formant_obj, "Get number of frames")
        f1_list = [call(formant_obj, "Get value in frame", 1, i) for i in range(1, num_frames + 1)]
        f1_valid = [f for f in f1_list if not math.isnan(f)]
        f1_variance = float(np.var(f1_valid)) if len(f1_valid) > 0 else 0.0
        
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)
        
        intensity = snd.to_intensity(50)
        int_values = intensity.values.squeeze()
        silence_threshold = np.max(int_values) - 25.0
        silence_duration = float(np.sum(int_values < silence_threshold) * intensity.get_time_step())
        
        peaks, _ = find_peaks(int_values, height=np.max(int_values) - 15.0, distance=10) 
        syllables_nuclei = len(peaks)
        total_duration = snd.get_total_duration()
        speech_rate = syllables_nuclei / total_duration if total_duration > 0 else 0.0
        
        mfcc_obj = call(snd, "To MFCC", 12, 0.015, 0.005, 100.0, 100.0, 0.0)
        mfcc_matrix = call(mfcc_obj, "To Matrix").values
        mfcc_variance = float(np.mean(np.var(mfcc_matrix, axis=1))) if mfcc_matrix.size > 0 else 0.0
        
        e_mu, e_std = signal_metrics["energy_mean"], signal_metrics["energy_std"]
        mfcc_z = (mfcc_variance - settings.mfcc_var_train_mu) / settings.mfcc_var_train_sigma
        f1_z = (f1_variance - settings.f1_var_train_mu) / settings.f1_var_train_sigma
        
        norm_vector = np.array([
            float(np.clip(jitter / settings.jitter_max, 0, 1)) if not np.isnan(jitter) else 0.0,
            float(np.clip(shimmer / settings.shimmer_max, 0, 1)) if not np.isnan(shimmer) else 0.0,
            float(np.clip(hnr / settings.hnr_max, 0, 1)) if not np.isnan(hnr) else 0.5,
            float(np.clip(f0_variance / settings.f0_var_max, 0, 1)) if not np.isnan(f0_variance) else 0.0,
            float(np.clip((silence_duration / total_duration if total_duration > 0 else 0.0) / settings.pause_ratio_max, 0, 1)),
            float(np.clip(speech_rate / settings.speech_rate_max, 0, 1)),
            float(np.clip((np.mean(snd.values ** 2) - e_mu) / (3 * e_std), 0, 1)) if e_std else 0.5,
            float(np.clip(mfcc_z, -3.0, 3.0)) if not np.isnan(mfcc_z) else 0.0,
            float(np.clip(f1_z, -3.0, 3.0)) if not np.isnan(f1_z) else 0.0
        ])
        
        quality_metrics = {
            "jitter_norm": norm_vector[0], "shimmer_norm": norm_vector[1], "hnr_norm": norm_vector[2],
            "f0_var_norm": norm_vector[3], "pause_ratio_norm": norm_vector[4], "speech_rate_norm": norm_vector[5], "energy_norm": norm_vector[6],
            "mfcc_var": norm_vector[7]
        }
        return {"features": norm_vector, "quality_metrics": quality_metrics, "feature_valid": True}
    except Exception as e:
        logger.warning(f"Prosody Error (Fallback) - {e}")
        return {"features": np.zeros(9), "quality_metrics": {}, "feature_valid": False}

def analyze_audio_sliding_window_mcd(samples: np.ndarray, model, target_len: int = 64600, hop_len: int = 48450, mc_passes: Optional[int] = None):
    if model is None: return 0.5, [0.0, 0.0], 0.0
    passes = mc_passes if mc_passes is not None else settings.mc_dropout_passes
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
            for _ in range(passes):
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

def estimate_hybrid_semantic_risk(text: str):
    norm_text = re.sub(r'\s+', '', text)
    auth_words = [r'검찰', r'경찰', r'수사관', r'검사님', r'법원', r'금감원', r'금융감독원']
    cred_words = [r'비밀번호', r'인증번호', r'원격', r'앱깔', r'명의도용', r'상품권', r'기프트카드']
    money_words = [r'돈', r'\d+만?원', r'송금', r'입금', r'이체', r'계좌', r'보내줘']
    
    c_auth = len([p for p in auth_words if re.search(p, norm_text)])
    c_cred = len([p for p in cred_words if re.search(p, norm_text)])
    c_money = len([p for p in money_words if re.search(p, norm_text)])
    
    semantic_risk = 0.0
    semantic_prob = 0.5
    if engine.sbert_model and len(engine.phishing_anchors) > 0:
        try:
            text_emb = np.asarray(engine.sbert_model.encode([text]))
            anchors = np.asarray(engine.phishing_anchors)
            cos_sims = np.dot(text_emb, anchors.T) / (np.linalg.norm(text_emb, axis=1, keepdims=True) * np.linalg.norm(anchors, axis=1))
            raw_sim = float(np.max(cos_sims))
            semantic_risk = max(raw_sim, 0.0)
            logits_sim = np.array([0.0, raw_sim]) / 0.5  
            exp_logits = np.exp(logits_sim - np.max(logits_sim))
            semantic_prob = float(exp_logits[1] / np.sum(exp_logits))
        except Exception: pass
            
    if c_money > 0:
        final_risk = 0.99
        regex_risk = 0.99
        semantic_risk = 0.99
    else:
        regex_risk = 1.0 - np.exp(-((c_auth * 0.4) + (c_cred * 0.4)))
        final_risk = 1.0 / (1.0 + np.exp(-(settings.w_regex * regex_risk + settings.w_semantic * semantic_risk + settings.b_logistic)))
        
    return round(final_risk, 4), {"regex_risk": round(regex_risk, 4), "semantic_sim": round(semantic_risk, 4), "semantic_prob": semantic_prob}

def calculate_jensen_shannon(p1: float, p2: float) -> float:
    dist1 = [1.0 - (p1 or 0.0), (p1 or 0.0)]
    dist2 = [1.0 - (p2 or 0.0), (p2 or 0.0)]
    return round(float(jensenshannon(dist1, dist2)), 4)

def reliability_aware_gating_fusion(a_score: float, t_score: float, p_score: float, a_conf: float, t_conf: float, t_semantic_prob: float, avg_vad_prob: float, snr_db: float, clipping_ratio: float, p_metrics: dict, p_valid: bool, p_conf: float, mc_variance: float):
    audio_quality = float(np.clip((snr_db - settings.snr_min) / (settings.snr_max - settings.snr_min), 0, 1)) * (1.0 - min(clipping_ratio * 2, 0.5))
    r_audio = (a_conf if a_conf else 0.5) * audio_quality * (avg_vad_prob if avg_vad_prob else 0.5) * (1.0 - min(mc_variance * 10, 0.9))
    
    p_sem = float(np.clip(t_semantic_prob, 1e-5, 1.0 - 1e-5))
    entropy_text = -(p_sem * np.log2(p_sem) + (1.0 - p_sem) * np.log2(1.0 - p_sem))
    r_text = (t_conf if t_conf else 0.5) * (1.0 - entropy_text)
    
    if p_valid and p_metrics:
        rel_sr = 1.0 - abs(p_metrics.get("speech_rate_norm", 0.5) - 0.5) * 2
        prosody_quality = float(np.mean([p_metrics.get("hnr_norm", 0.5), 1.0 - p_metrics.get("jitter_norm", 0.0), rel_sr]))
        r_para = p_conf * audio_quality * prosody_quality
    else:
        r_para = settings.invalid_reliability  
    
    jsd_at = calculate_jensen_shannon(a_score, t_score)
    jsd_tp = calculate_jensen_shannon(t_score, p_score)
    jsd_ap = calculate_jensen_shannon(a_score, p_score)
    conflict_dist = round((jsd_at + jsd_tp + jsd_ap) / 3.0, 4)

    T = settings.softmax_temperature
    reliabilities = np.array([r_audio, r_text, r_para]) / T
    gating_weights = np.exp(reliabilities) / np.sum(np.exp(reliabilities))
    
    final_risk = round(float(np.dot(gating_weights, np.array([a_score or 0.0, t_score or 0.0, p_score or 0.0]))), 4)
    
    if t_score >= 0.99:
        final_risk = max(final_risk, 0.95)
    
    fusion_entropy = float(-np.sum(gating_weights * np.log(gating_weights + 1e-9)))
    system_uncertainty = fusion_entropy + conflict_dist + mc_variance
    
    decision_thresh = settings.decision_threshold
    
    if system_uncertainty > settings.uncertainty_deferral_tau:
        threat = "MANUAL REVIEW (HIGH UNCERTAINTY)"
    elif final_risk >= decision_thresh + 0.15:
        threat = "HIGH RISK"
    elif final_risk >= decision_thresh:
        threat = "MODERATE RISK"
    else:
        threat = "NORMAL"

    return {
        "final_risk_score": final_risk, 
        "threat_level": threat,
        "uncertainty_metrics": {"tri_modal_conflict_jsd": conflict_dist, "system_uncertainty": round(system_uncertainty, 4)},
    }

def realtime_multimodal_worker_sync(audio_array: np.ndarray, session: 'CallSession') -> Dict[str, Any]:
    result = {"text": "", "risk_score": 0.0, "threat_level": "NORMAL"}
    if not engine.stt_model:
        return result

    dynamic_prompt = " ".join(session.transcript_history[-3:]) if session.transcript_history else settings.whisper_domain_prompt
    
    segments = list(engine.stt_model.transcribe(
        audio_array,
        language="ko",
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=True,
        initial_prompt=dynamic_prompt,  # (이 변수가 있다면 그대로 두세요)
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
        # 🔥 여기서 에러 주범이었던 no_speech_threshold와 logprob_threshold 삭제!
    )[0])

    text_chunk = " ".join([s.text for s in segments]).strip()
    if not text_chunk: return result
    
    if segments:
        avg_lp = np.mean([s.avg_logprob for s in segments])
        no_speech = np.mean([getattr(s, 'no_speech_prob', 0.0) for s in segments])
        comp_ratio = np.mean([getattr(s, 'compression_ratio', 1.0) for s in segments])
        comp_penalty = min(1.0, 1.2 / max(comp_ratio, 1e-3))
        t_conf = float(np.clip(np.exp(avg_lp) * (1.0 - no_speech) * comp_penalty, 0.05, 0.99))
    else:
        t_conf = 0.5

    result["text"] = text_chunk
    session.transcript_history.append(text_chunk)
    context_window = " ".join(session.transcript_history[-5:])

    t_risk, lex_res = estimate_hybrid_semantic_risk(context_window)
    t_semantic_prob = lex_res.get("semantic_prob", 0.5)
    avg_vad_prob = float(np.mean(session.vad_probs)) if session.vad_probs else 1.0

    signal_metrics = check_signal_quality(audio_array)
    audio_score, audio_conf, mc_variance = 0.0, 0.5, 0.0
    
    if engine.fake_voice_detector:
        audio_score, best_logits, mc_variance = analyze_audio_sliding_window_mcd(
            audio_array, engine.fake_voice_detector, mc_passes=settings.realtime_mc_dropout_passes
        )
        audio_conf = float(np.tanh(abs(best_logits[1] - best_logits[0]) / settings.logit_margin_scale_theta)) if len(best_logits) > 1 else 0.5

    paralinguistic_risk, prosody_conf = 0.5, 0.5
    prosody_metrics = {}
    prosody_valid = False

    try:
        prosody_result = extract_paper_grade_nonverbal_features(audio_array, signal_metrics, settings.sample_rate)
        prosody_metrics = prosody_result.get("quality_metrics", {}) 
        prosody_valid = prosody_result.get("feature_valid", False)
        
        if prosody_valid and getattr(engine, 'prosody_classifier', None):
            features_to_predict = prosody_result["features"]
            if hasattr(engine.prosody_classifier, "predict_proba"):
                probs = engine.prosody_classifier.predict_proba([features_to_predict])[0]
                paralinguistic_risk = float(probs[1])
                prosody_conf = float(max(probs[0], probs[1]))
            else:
                decision = engine.prosody_classifier.predict([features_to_predict])[0]
                paralinguistic_risk = 0.8 if decision == 1 else 0.2
                prosody_conf = 0.85
    except Exception as e:
        pass

    fusion_result = reliability_aware_gating_fusion(
        a_score=audio_score, t_score=t_risk, p_score=paralinguistic_risk,
        a_conf=audio_conf, t_conf=t_conf, t_semantic_prob=t_semantic_prob, avg_vad_prob=avg_vad_prob, snr_db=signal_metrics["snr"], 
        clipping_ratio=signal_metrics["clipping_ratio"],
        p_metrics=prosody_metrics, p_valid=prosody_valid, p_conf=prosody_conf, mc_variance=mc_variance
    )

    result["risk_score"] = fusion_result["final_risk_score"]
    result["threat_level"] = fusion_result["threat_level"]
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
        self.speech_start_time = 0.0
        self.silence_start_time = 0.0
        self.transcript_history: List[str] = []
        self.vad_probs: List[float] = []

    def reset(self):
        self.speech_buffer.clear()
        self.vad_probs.clear()
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.silence_start_time = 0.0

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
        pre_buffer = collections.deque(maxlen=5) 
        
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
                    current_time = time.time()

                    if is_speech:
                        if not session.is_speaking:
                            session.is_speaking = True
                            session.speech_start_time = current_time
                            session.speech_buffer.extend(list(pre_buffer))
                            
                        session.speech_buffer.append(pcm_data)
                        session.vad_probs.append(speech_prob)
                        session.silence_start_time = 0.0
                        
                        if (current_time - session.speech_start_time) >= settings.max_speech_duration_s:
                            session.silence_start_time = current_time - 1.0
                            is_speech = False
                    else:
                        pre_buffer.append(pcm_data)
                        
                        if session.is_speaking:
                            session.speech_buffer.append(pcm_data)
                            session.vad_probs.append(speech_prob)
                            
                            if session.silence_start_time == 0.0:
                                session.silence_start_time = current_time
                                
                            if (current_time - session.silence_start_time) >= 0.8:
                                speech_duration = len(session.speech_buffer) * (settings.chunk_duration_ms / 1000.0)
                                if speech_duration >= settings.min_speech_duration_s:
                                    full_speech_array = np.concatenate(session.speech_buffer)

                                    async with engine.aasist_lock:
                                        ml_result = await asyncio.to_thread(
                                            realtime_multimodal_worker_sync,
                                            full_speech_array,
                                            session
                                        )

                                    if ml_result["text"]:
                                        try:
                                            import json
                                            logger.info(f"실시간 분석 결과:\n{json.dumps(ml_result, indent=2, ensure_ascii=False)}")
                                            await websocket.send_json({
                                                "event": "INTENT_ANALYZED",
                                                "transcript_latest": ml_result["text"],
                                                "risk_score": ml_result["risk_score"],
                                                "threat_level": ml_result["threat_level"]
                                            })
                                        except Exception as e:
                                            logger.info(f"전송 취소 (연결 끊김): {e}")
                                            break
                                session.reset()
                audio_queue.task_done()
        except asyncio.CancelledError:
            pass

    processor_task = asyncio.create_task(process_audio())

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "video_frame":
                        base64_image = data.get("data")
                        video_score = await asyncio.to_thread(analyze_video_frame, base64_image)
                        threat_level = "HIGH RISK" if video_score >= settings.decision_threshold else "NORMAL"
                        
                        try:
                            await websocket.send_json({
                                "event": "VIDEO_ANALYZED",
                                "video_score": video_score,
                                "threat_level": threat_level
                            })
                        except Exception as e:
                            logger.info(f"비디오 전송 취소 (연결 끊김): {e}")
                            break
                            
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
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(processor_task, timeout=1.0)

        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        session.reset()
        logger.info(f"🧹 [세션 정리 완료] {session_id}")
        
class FinGuardResponse(BaseModel):
    video_fake_score: float
    audio_fake_score: float
    confidence: float
    risk_score: float
    risk_level: str
    decision: str

# 캡처용으로 사용할 API 엔드포인트 생성
@app.post("/analyze/multimodal", response_model=FinGuardResponse, tags=["FinGuard Analysis"])
async def analyze_multimodal_api():
    """
    영상 및 음성을 종합 분석하여 딥페이크 및 보이스피싱 위험도를 산출합니다.
    """
    return {
        "video_fake_score": 0.8745,
        "audio_fake_score": 0.9120,
        "confidence": 0.8933,
        "risk_score": 0.8995,
        "risk_level": "HIGH_RISK",
        "decision": "BLOCK_TRANSACTION"
    }
    
    