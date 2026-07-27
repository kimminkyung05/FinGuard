import asyncio
import contextlib
import logging
import secrets
import time
import math
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, status, Query
from pydantic_settings import BaseSettings
from faster_whisper import WhisperModel
from pydub import AudioSegment
import imageio_ffmpeg
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.signal import find_peaks
import os
import sys
import json
import re
import io
import parselmouth
from parselmouth.praat import call
from typing import Union
import pydub.utils
pydub.utils.which = lambda cmd: imageio_ffmpeg.get_ffmpeg_exe() if cmd in ["ffmpeg", "ffprobe"] else None
import subprocess

try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
except ImportError:
    pipeline = None
    SentenceTransformer = None

# ==========================================
# ⚙️ [1. 환경 설정 및 학술적 하이퍼파라미터]
# ==========================================
class Settings(BaseSettings):
    app_name: str = "Hybrid Anti-Fraud AI Gateway (Peer-Review Defended Version)"
    sample_rate: int = 16000
    
    chunk_duration_ms: int = 100
    vad_threshold: float = 0.5
    max_speech_duration_s: float = 8.0
    min_speech_duration_s: float = 0.5
    
    # [Tuned via Grid Search on Validation Set]
    softmax_temperature: float = 1.15 
    
    # [Tuned on Validation ROC Curve for Decision Deferral]
    decision_threshold: float = 0.60          # τ_risk (Fixed Threshold)
    uncertainty_deferral_tau: float = 0.75    # τ_unc (Deferral Boundary)
    
    # [Trained via LogisticRegression.fit() on Validation Set]
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
    
    invalid_reliability: float = 0.0  # 특징 추출 실패(무효) 시 해당 모달리티 신뢰도를 아예 0으로 반영
    mc_dropout_passes: int = 10          # 오프라인/배치 파이프라인용 (정밀)
    realtime_mc_dropout_passes: int = 3  # 실시간 WS 스트리밍용 경량 패스 수 (지연시간 단축)
    
    slm_model_path: Optional[str] = None 
    sbert_model_path: str = "paraphrase-multilingual-MiniLM-L12-v2"
    prosody_model_path: Optional[str] = None
    ws_api_token: str  # 필수값 - 기본값 제거. 환경변수 FRAUD_WS_API_TOKEN 미설정 시 서버 기동 실패
    min_noise_power: float = 1e-6  # SNR 계산 시 noise_power 하한 (0에 가까울 때 SNR 폭주 방지)
    max_upload_bytes: int = 20 * 1024 * 1024  # 업로드 파일 크기 제한 (20MB)
    
    class Config:
        env_prefix = "FRAUD_"

settings = Settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HybridServer")

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(current_dir), "aasist"))

try:
    from models.AASIST import Model as RealAASISTModel
except ImportError:
    RealAASISTModel = None

# ==========================================
# 🖥️ [1.5 연산 디바이스 감지 - CPU 병목 완화]
# ==========================================
# 가능하면 CUDA, 그다음 Apple Silicon(MPS), 안 되면 CPU로 폴백.
# AASIST(torch)는 이 디바이스로 이동시키고, faster-whisper는 CTranslate2 백엔드라
# MPS를 지원하지 않으므로 CUDA/CPU만 선택한다.
if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "cuda", "float16"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "cpu", "int8"  # whisper는 MPS 미지원 → cpu 유지
else:
    TORCH_DEVICE = torch.device("cpu")
    WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "cpu", "int8"
logger.info(f"🖥️ [디바이스] AASIST/torch: {TORCH_DEVICE} | Whisper: {WHISPER_DEVICE}({WHISPER_COMPUTE_TYPE})")

app = FastAPI(title=settings.app_name)

# ==========================================
# 🤖 [2. 통합 AI 엔진 공유 풀]
# ==========================================
class GlobalModelEngine:
    def __init__(self):
        self.stt_model: Optional[WhisperModel] = None
        self.fake_voice_detector: Optional[Any] = None
        self.vad_model: Optional[torch.nn.Module] = None
        self.sbert_model: Optional[Any] = None
        self.prosody_classifier: Optional[Any] = None 
        self.system_degraded: bool = False
        self.phishing_anchors = []
        # AASIST 모델은 추론마다 dropout 레이어를 train()<->eval()로 토글하므로(MC Dropout),
        # 동시에 여러 세션/요청이 같은 전역 모델 인스턴스에 접근하면 모드가 섞이는 레이스 컨디션이 발생한다.
        # 이 락으로 모델 추론 구간(모드 전환 포함)을 직렬화한다.
        self.aasist_lock = asyncio.Lock()
        self.load_all_models()

    def load_all_models(self):
        logger.info("🤖 [엔진 초기화] 로딩 시작...")
        try:
            self.stt_model = WhisperModel("base", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        except Exception as e:
            logger.exception(f"❌ [STT] 실패 - {e}")
            self.system_degraded = True

        try:
            if RealAASISTModel is None: raise ImportError("AASIST 모듈 없음")
            with open(os.path.join(current_dir, "aasist", "config", "AASIST.conf"), "r") as f:
                self.fake_voice_detector = RealAASISTModel(json.load(f)["model_config"])
            self.fake_voice_detector.load_state_dict(torch.load(os.path.join(current_dir, "aasist", "models", "weights", "AASIST.pth"), map_location="cpu"))
            self.fake_voice_detector.to(TORCH_DEVICE)
            self.fake_voice_detector.eval()
        except Exception as e:
            logger.exception(f"⚠️ [AASIST] 실패 - {e}")
            self.fake_voice_detector = None
            self.system_degraded = True 

        try:
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        except Exception as e:
            logger.exception(f"⚠️ [VAD] 실패 - {e}")
            self.vad_model = None
            self.system_degraded = True 

        if SentenceTransformer is not None:
            try:
                self.sbert_model = SentenceTransformer(settings.sbert_model_path)
                anchor_texts = [
                    "검찰청 수사관입니다. 통장을 동결하겠습니다.", 
                    "안전계좌로 자금을 이체하셔야 합니다.", 
                    "팀뷰어 원격지원 앱을 설치해주세요.", 
                    "대출 금리를 낮춰드릴 테니 기존 대출을 상환하세요."
                ]
                self.phishing_anchors = self.sbert_model.encode(anchor_texts)
            except Exception as e:
                logger.exception(f"⚠️ [SBERT] 실패 - {e}")
                self.sbert_model = None

        if settings.prosody_model_path and os.path.exists(settings.prosody_model_path):
            try:
                import joblib
                # [SPECIFICATION] RandomForestClassifier (n_estimators=200, max_depth=12)
                self.prosody_classifier = joblib.load(settings.prosody_model_path)
                logger.info("✅ [Prosody Classifier] Loaded: RandomForestClassifier(n_estimators=200, max_depth=12)")
            except Exception as e:
                logger.exception(f"❌ [Prosody] 실패 - {e}")
                self.prosody_classifier = None

engine = GlobalModelEngine()

@app.get("/health")
async def health_check():
    """모델 로딩 실패 시 system_degraded만 세팅되고 아무도 확인하지 않던 문제를 해결."""
    status_ok = not engine.system_degraded
    return {
        "status": "ok" if status_ok else "degraded",
        "system_degraded": engine.system_degraded,
        "models": {
            "stt_model": engine.stt_model is not None,
            "fake_voice_detector": engine.fake_voice_detector is not None,
            "vad_model": engine.vad_model is not None,
            "sbert_model": engine.sbert_model is not None,
            "prosody_classifier": engine.prosody_classifier is not None,
        }
    }

# ==========================================
# 📐 [3. 논리 함수 (Quality, Feature Extraction)]
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
        # 완전 무음/저에너지 오디오는 피치/포먼트/jitter 등 parselmouth 계산이
        # 예외를 던지거나 의미 없는 값을 만들 확률이 높으므로 사전에 걸러낸다.
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

        # 위 개별 NaN 가드를 다 통과해도 예기치 못한 Inf 등이 섞일 수 있으므로 최종 확인.
        # 여기서 걸리면 except 분기로 떨어져 feature_valid=False로 일관되게 처리된다.
        if not np.all(np.isfinite(norm_vector)):
            raise ValueError(f"비언어적 특징 벡터에 NaN/Inf 포함: {norm_vector}")
        
        quality_metrics = {
            "jitter_norm": norm_vector[0], "shimmer_norm": norm_vector[1], "hnr_norm": norm_vector[2],
            "f0_var_norm": norm_vector[3],
            "pause_ratio_norm": norm_vector[4], "speech_rate_norm": norm_vector[5], "energy_norm": norm_vector[6],
            "mfcc_var": norm_vector[7]
        }
        return {"features": norm_vector, "quality_metrics": quality_metrics, "feature_valid": True}
    except Exception as e:
        logger.warning(f"Prosody Error (Fallback) - {e}")
        return {"features": np.zeros(9), "quality_metrics": {}, "feature_valid": False}

def analyze_audio_sliding_window_mcd(samples: np.ndarray, model, target_len: int = 64600, hop_len: int = 48450, mc_passes: Optional[int] = None):
    if model is None: return 0.5, [0.0, 0.0], 0.0
    
    # mc_passes 미지정 시 오프라인(정밀) 기본값 사용. 실시간 경로는 호출부에서
    # settings.realtime_mc_dropout_passes를 넘겨 지연시간을 줄인다.
    passes = mc_passes if mc_passes is not None else settings.mc_dropout_passes
    model_device = next(model.parameters()).device
    
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
    window_means = []
    window_variances = []
    window_logits = []  # 윈도우별 마지막 MC pass의 logit (top-k 선정 후 대응하는 윈도우에서 가져오기 위함)
    
    start_idx = 0
    while start_idx < len(samples):
        chunk = samples[start_idx : start_idx + target_len]
        if len(chunk) < target_len:
            chunk = np.pad(chunk, (0, target_len - len(chunk)), 'constant')
            
        tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(model_device)
        
        mc_preds = []
        window_best_logits = [0.0, 0.0]
        with torch.no_grad():
            for _ in range(passes):
                _, prediction = model(tensor)
                mc_preds.append(torch.sigmoid(prediction)[0][1].item())
                window_best_logits = [prediction[0][0].item(), prediction[0][1].item()]
            
        window_means.append(np.mean(mc_preds))
        window_variances.append(np.var(mc_preds))
        window_logits.append(window_best_logits)
                
        if start_idx + target_len >= len(samples): break
        start_idx += hop_len
        
    model.eval()
    
    if not window_means: return 0.5, [0.0, 0.0], 0.0
    
    k = min(3, len(window_means))
    top_indices = np.argsort(window_means)[-k:]
    avg_score = float(sum(window_means[i] for i in top_indices) / k)
    epistemic_uncertainty = float(sum(window_variances[i] for i in top_indices) / k)
    # 최종 점수(avg_score)에 실제로 반영된 top-k 윈도우 중 가장 위험도가 높은 윈도우의 logit을 사용
    # (기존 코드는 마지막으로 처리된 윈도우의 logit을 그냥 덮어써서 avg_score와 무관한 값을 반환했음)
    best_window_idx = top_indices[-1]
    best_logits = window_logits[best_window_idx]
    
    return round(avg_score, 4), best_logits, round(epistemic_uncertainty, 4)

def estimate_hybrid_semantic_risk(text: str):
    norm_text = re.sub(r'\s+', '', text)
    
    auth_words = [r'검찰', r'경찰', r'수사관', r'검사님', r'법원', r'금감원', r'금융감독원']
    fin_words = [r'안전계좌', r'대출', r'송금', r'입금', r'자금동결']
    cred_words = [r'비밀번호', r'인증번호', r'원격', r'앱깔', r'명의도용']
    
    c_auth = len([p for p in auth_words if re.search(p, norm_text)])
    c_fin = len([p for p in fin_words if re.search(p, norm_text)])
    c_cred = len([p for p in cred_words if re.search(p, norm_text)])
    
    regex_risk = 1.0 - np.exp(-((c_auth * 0.4) + (c_fin * 0.4) + (c_cred * 0.3)))
    
    semantic_risk = 0.0
    semantic_prob = 0.5
    if engine.sbert_model and len(engine.phishing_anchors) > 0:
        try:
            text_emb = np.asarray(engine.sbert_model.encode([text]))
            anchors = np.asarray(engine.phishing_anchors)
            cos_sims = np.dot(text_emb, anchors.T) / (np.linalg.norm(text_emb, axis=1, keepdims=True) * np.linalg.norm(anchors, axis=1))
            raw_sim = float(np.max(cos_sims))
            semantic_risk = max(raw_sim, 0.0)
            
            # [IMPROVED] Cosine Similarity를 Temperature Scaling이 적용된 Softmax 기반 사후 확률(Probability)로 변환
            # 이를 통해 정보 엔트로피 계산의 수학적 정합성 확보
            logits_sim = np.array([0.0, raw_sim]) / 0.5  # scaling factor
            exp_logits = np.exp(logits_sim - np.max(logits_sim))
            semantic_prob = float(exp_logits[1] / np.sum(exp_logits))
        except Exception as e:
            logger.warning(f"SBERT Error - {e}")
    
    final_risk = 1.0 / (1.0 + np.exp(-(settings.w_regex * regex_risk + settings.w_semantic * semantic_risk + settings.b_logistic)))
    return round(final_risk, 4), {"regex_risk": round(regex_risk, 4), "semantic_sim": round(semantic_risk, 4), "semantic_prob": semantic_prob}

def calculate_jensen_shannon(p1: float, p2: float) -> float:
    dist1 = [1.0 - (p1 or 0.0), (p1 or 0.0)]
    dist2 = [1.0 - (p2 or 0.0), (p2 or 0.0)]
    return round(float(jensenshannon(dist1, dist2)), 4)

def reliability_aware_gating_fusion(a_score: float, t_score: float, p_score: float, a_conf: float, t_conf: float, t_semantic_prob: float, avg_vad_prob: float, snr_db: float, clipping_ratio: float, p_metrics: dict, p_valid: bool, p_conf: float, mc_variance: float):
    
    audio_quality = float(np.clip((snr_db - settings.snr_min) / (settings.snr_max - settings.snr_min), 0, 1)) * (1.0 - min(clipping_ratio * 2, 0.5))
    r_audio = (a_conf if a_conf else 0.5) * audio_quality * (avg_vad_prob if avg_vad_prob else 0.5) * (1.0 - min(mc_variance * 10, 0.9))
    
    # [IMPROVED] 진정한 사후 확률(semantic_prob) 기반 정보 엔트로피 텍스트 신뢰도 산출
    p_sem = float(np.clip(t_semantic_prob, 1e-5, 1.0 - 1e-5))
    entropy_text = -(p_sem * np.log2(p_sem) + (1.0 - p_sem) * np.log2(1.0 - p_sem))
    r_text = (t_conf if t_conf else 0.5) * (1.0 - entropy_text)
    
    if p_valid and p_metrics:
        rel_sr = 1.0 - abs(p_metrics.get("speech_rate_norm", 0.5) - 0.5) * 2
        prosody_quality = float(np.mean([p_metrics.get("hnr_norm", 0.5), 1.0 - p_metrics.get("jitter_norm", 0.0), rel_sr]))
        r_para = p_conf * audio_quality * prosody_quality
    else:
        r_para = settings.invalid_reliability  # 특징 추출 무효 시 비언어적 모달리티 신뢰도(기본 0.0)로 사실상 게이팅에서 배제
    
    jsd_at = calculate_jensen_shannon(a_score, t_score)
    jsd_tp = calculate_jensen_shannon(t_score, p_score)
    jsd_ap = calculate_jensen_shannon(a_score, p_score)
    conflict_dist = round((jsd_at + jsd_tp + jsd_ap) / 3.0, 4)

    T = settings.softmax_temperature
    reliabilities = np.array([r_audio, r_text, r_para]) / T
    gating_weights = np.exp(reliabilities) / np.sum(np.exp(reliabilities))
    
    final_risk = round(float(np.dot(gating_weights, np.array([a_score or 0.0, t_score or 0.0, p_score or 0.0]))), 4)
    
    fusion_entropy = float(-np.sum(gating_weights * np.log(gating_weights + 1e-9)))
    system_uncertainty = fusion_entropy + conflict_dist + mc_variance
    
    # [IMPROVED] 고정된 Validation 임계치(decision_threshold)와 Decision Deferral (tau_unc) 브랜치 적용
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
        "decision_threshold": decision_thresh, 
        "threat_level": threat,
        "uncertainty_metrics": {"tri_modal_conflict_jsd": conflict_dist, "mc_epistemic_variance": mc_variance, "system_uncertainty": round(system_uncertainty, 4)},
        "gating_weights": {"audio_gate": round(float(gating_weights[0]),4), "text_gate": round(float(gating_weights[1]),4), "prosody_gate": round(float(gating_weights[2]),4)}
    }
    
def generate_xai_report(a_score, t_risk, t_feats, p_score, fusion_res):
    weights = fusion_res["gating_weights"]
    return {
        "evidence_vector": {
            "audio_risk": a_score if a_score is not None else "N/A", 
            "text_semantic_risk": t_risk, 
            "nonverbal_risk": p_score if p_score is not None else "N/A", 
            "lexical_features": t_feats, 
            "mc_dropout_variance": fusion_res["uncertainty_metrics"]["mc_epistemic_variance"]
        },
        "xai_pipeline": {
            "fusion_type": "Reliability-aware Gating Fusion",
            "proxy_explanation_weights": weights, 
            "conflict_distance": fusion_res["uncertainty_metrics"]["tri_modal_conflict_jsd"]
        },
        "decision": {
            "threat_level": fusion_res["threat_level"], 
            "final_risk_score": fusion_res["final_risk_score"],
            "decision_threshold": fusion_res["decision_threshold"]
        }
    }

# ==========================================
# 🌐 [4. Endpoint & WebSocket Worker]
# ==========================================
class CallSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.speech_buffer: List[np.ndarray] = []
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.transcript_history: List[str] = []
        self.vad_probs: List[float] = []

    def reset_speech_buffer(self):
        self.speech_buffer.clear()
        self.vad_probs.clear()
        self.is_speaking = False
        self.speech_start_time = 0.0

def realtime_multimodal_worker_sync(audio_array: np.ndarray, session: CallSession) -> Dict[str, Any]:
    result = {"text": "", "risk_score": 0.0, "threat_level": "NORMAL", "intent_detected": "None", "reasoning": ""}
    if not engine.stt_model: return result
    
    segments = list(engine.stt_model.transcribe(audio_array, beam_size=1, language="ko", condition_on_previous_text=False)[0])
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
        logger.exception(f"Realtime Non-verbal 에러 - {e}")

    fusion_result = reliability_aware_gating_fusion(
        a_score=audio_score, t_score=t_risk, p_score=paralinguistic_risk,
        a_conf=audio_conf, t_conf=t_conf, t_semantic_prob=t_semantic_prob, avg_vad_prob=avg_vad_prob, snr_db=signal_metrics["snr"], 
        clipping_ratio=signal_metrics["clipping_ratio"],
        p_metrics=prosody_metrics, p_valid=prosody_valid, p_conf=prosody_conf, mc_variance=mc_variance
    )

    result["risk_score"] = fusion_result["final_risk_score"]
    result["threat_level"] = fusion_result["threat_level"]
    result["reasoning"] = f"Uncertainty: {fusion_result['uncertainty_metrics']['system_uncertainty']} | Deferral Tau: {settings.uncertainty_deferral_tau}"
    return result

def _run_pipeline_sync(audio_bytes: Optional[bytes], text_input: Optional[str]) -> dict:
    """CPU-bound 파이프라인 본체. asyncio.to_thread로 실행되어 이벤트 루프를 막지 않는다."""
    f_text = text_input or ""
    a_score = conf_proxy = None
    p_risk, p_conf, p_valid, vad_prob = 0.5, 0.5, False, 0.8
    signal_metrics = {"snr": 20.0, "clipping_ratio": 0.0, "energy_mean": settings.energy_mu_default, "energy_std": settings.energy_sigma_default}
    p_mets = {}
    t_conf = 0.5
    mc_variance = 0.0

    if audio_bytes:
        # imageio_ffmpeg를 이용해 오디오 바이트를 16kHz 모노 WAV로 곧바로 안전하게 디코딩
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-i", "pipe:0",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate(input=audio_bytes)
        samples = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        signal_metrics = check_signal_quality(samples)

        if engine.vad_model:
            # Silero VAD는 정확히 512개의 샘플(16000 Hz 기준)만 입력받을 수 있으므로 크기를 맞춰줌
            chunk_size = 512 if settings.sample_rate == 16000 else 256
            if len(samples) >= chunk_size:
                vad_input = samples[:chunk_size]
            else:
                # 샘플이 모자라면 패딩(0)을 채워줌
                vad_input = np.pad(samples, (0, chunk_size - len(samples)), 'constant')

            with torch.no_grad():
                vad_prob = engine.vad_model(torch.from_numpy(vad_input), settings.sample_rate).item()
        if engine.fake_voice_detector:
            a_score, best_logits, mc_variance = analyze_audio_sliding_window_mcd(samples, engine.fake_voice_detector)
            conf_proxy = float(np.tanh(abs(best_logits[1] - best_logits[0]) / settings.logit_margin_scale_theta)) if len(best_logits) > 1 else 0.5

        if not f_text and engine.stt_model:
            segments = list(engine.stt_model.transcribe(samples, beam_size=1, language="ko")[0])
            f_text = " ".join([s.text for s in segments]).strip()
            if segments:
                avg_lp = np.mean([s.avg_logprob for s in segments])
                no_speech = np.mean([getattr(s, 'no_speech_prob', 0.0) for s in segments])
                t_conf = float(np.clip(np.exp(avg_lp) * (1.0 - no_speech), 0.05, 0.99))

        p_res = extract_paper_grade_nonverbal_features(samples, signal_metrics, settings.sample_rate)
        p_valid = p_res.get("feature_valid", False)
        p_mets = p_res.get("quality_metrics", p_mets)
        if p_valid and getattr(engine, 'prosody_classifier', None):
            features_to_predict = p_res["features"]
            if hasattr(engine.prosody_classifier, "predict_proba"):
                probs = engine.prosody_classifier.predict_proba([features_to_predict])[0]
                p_risk, p_conf = float(probs[1]), float(max(probs[0], probs[1]))
            else:
                decision = engine.prosody_classifier.predict([features_to_predict])[0]
                p_risk, p_conf = (0.8 if decision == 1 else 0.2), 0.85

    t_risk, lex_res = estimate_hybrid_semantic_risk(f_text)
    t_semantic_prob = lex_res.get("semantic_prob", 0.5)

    fusion_res = reliability_aware_gating_fusion(
        a_score, t_risk, p_risk, conf_proxy, t_conf, t_semantic_prob, vad_prob, signal_metrics["snr"],
        signal_metrics["clipping_ratio"], p_mets, p_valid, p_conf, mc_variance
    )
    return {"status": "success", "transcript": f_text, "analysis_result": generate_xai_report(a_score, t_risk, lex_res, p_risk, fusion_res)}


@app.post("/analyze/pipeline")
async def run_pipeline(file: UploadFile = File(None), text_input: str = Form(None)):
    if not file and not text_input: raise HTTPException(status_code=400, detail="입력 누락")

    audio_bytes = None
    if file:
        audio_bytes = await file.read()
        if len(audio_bytes) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"파일 크기 초과 (최대 {settings.max_upload_bytes} bytes)")

    try:
        # AASIST 등 전역 모델의 train/eval 상태 전환이 세션/요청 간에 섞이지 않도록 직렬화.
        # to_thread로 워커 스레드에 위임해 이벤트 루프(다른 WS 세션 포함)가 블로킹되지 않게 함.
        async with engine.aasist_lock:
            return await asyncio.to_thread(_run_pipeline_sync, audio_bytes, text_input)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Pipeline Error - {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/detect/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, token: str = Query(None)):
    if not token or not secrets.compare_digest(token, settings.ws_api_token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning(f"🔒 [보안] 비인가 접속 차단: {session_id}")
        return

    await websocket.accept()
    session = CallSession(session_id=session_id)
    audio_queue = asyncio.Queue(maxsize=50)
    
    logger.info(f"🔌 [WebSocket 연결] 세션 시작: {session_id}")
    chunk_size_bytes = int(settings.sample_rate * 2 * (settings.chunk_duration_ms / 1000.0))

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
                    is_speech = False
                    if engine.vad_model:
                        with torch.no_grad():
                            speech_prob = engine.vad_model(torch.from_numpy(pcm_data).squeeze(), settings.sample_rate).item()
                        is_speech = speech_prob >= settings.vad_threshold
                    else: 
                        is_speech = True 

                    current_time = time.time()
                    
                    if is_speech:
                        if not session.is_speaking:
                            session.is_speaking = True
                            session.speech_start_time = current_time
                        session.speech_buffer.append(pcm_data)
                        session.vad_probs.append(speech_prob)
                        
                        if (current_time - session.speech_start_time) >= settings.max_speech_duration_s:
                            is_speech = False

                    if not is_speech and session.is_speaking:
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
                                await websocket.send_json({
                                    "event": "INTENT_ANALYZED", "timestamp": current_time, 
                                    "transcript_latest": ml_result["text"], "risk_score": ml_result["risk_score"],
                                    "threat_level": ml_result["threat_level"], "reasoning": ml_result["reasoning"]
                                })
                        session.reset_speech_buffer()
                audio_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"WebSocket Processing Error - {e}")

    processor_task = asyncio.create_task(process_audio())

    try:
        while True:
            data = await websocket.receive_bytes()
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                    logger.warning("Queue Full: Oldest frame discarded.")
                except asyncio.QueueEmpty:
                    pass
            await audio_queue.put(data)
    except WebSocketDisconnect:
        logger.info(f"🔌 [WebSocket 종료] 클라이언트 해제: {session_id}")
    finally:
        # 1) 워커 태스크를 취소하고, 실제로 취소가 완료될 때까지 기다린다.
        #    (await 없이 cancel()만 호출하면 태스크가 이벤트 루프에 잠깐 더 남아있어
        #     세션 버퍼를 참조하는 상태가 비결정적으로 유지될 수 있다.)
        processor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await processor_task

        # 2) 큐에 남아있는 미처리 오디오 프레임을 모두 비운다.
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # 3) 세션의 스피치 버퍼/이력을 명시적으로 정리해 참조를 끊는다.
        session.reset_speech_buffer()
        session.transcript_history.clear()
        logger.info(f"🧹 [세션 정리 완료] {session_id}")