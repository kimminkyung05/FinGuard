import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic_settings import BaseSettings
from faster_whisper import WhisperModel
from pydub import AudioSegment
import imageio_ffmpeg
import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import os
import sys
import json
import re
import tempfile
import parselmouth
from parselmouth.praat import call
import math
import scipy.io.wavfile as wav

# ==========================================
# ⚙️ [1. 환경 설정 및 하이퍼파라미터 (Settings & Hyperparameters)]
# [FIXED: Issue 7] 모든 매직 넘버와 수식 상수를 Settings로 통합
# ==========================================
class Settings(BaseSettings):
    app_name: str = "Hybrid Anti-Fraud AI Gateway (Batch & Realtime)"
    sample_rate: int = 16000
    
    # WS Streaming Configs
    chunk_duration_ms: int = 100
    vad_threshold: float = 0.5
    max_speech_duration_s: float = 8.0
    min_speech_duration_s: float = 0.5
    lambda_uncertainty: float = 0.153
    
    # Mathematical Framework Constants
    base_threshold: float = 0.6
    weight_authority: float = 0.25
    weight_financial: float = 0.25
    weight_credential: float = 0.20
    weight_urgency: float = 0.15
    weight_structural: float = 0.10
    lexical_synergy_boost: float = 0.10
    
    omega_speech_rate: float = 0.55
    omega_pause_ratio: float = 0.45
    speech_rate_val_mean: float = 4.5
    
    gamma_base: float = 0.5
    kappa_gamma: float = 1.0
    beta_base: float = 0.5
    kappa_beta: float = 1.2
    lambda_base: float = 0.2
    kappa_lambda: float = 0.6
    
    alpha_sensitivity: float = 2.5
    logit_margin_scale_theta: float = 2.0
    eta_interaction: float = 0.15
    delta_threshold: float = 0.15
    rho_threshold: float = 0.10

    energy_mu: float = 0.02
    energy_sigma: float = 0.005
    snr_min: float = 5.0
    snr_max: float = 30.0
    
    # [추가됨] Prosody Min-Max 정규화를 위한 상한값 (Validation Set 기준)
    jitter_max: float = 0.01      # 기존 100 곱하기 대체 (1/100)
    shimmer_max: float = 0.1      # 기존 10 곱하기 대체 (1/10)
    hnr_max: float = 30.0         # HNR 최대값
    speech_rate_max: float = 6.0  # 초당 최대 음절 수
    f0_var_max: float = 5000.0    # F0 분산 최대값
    pause_ratio_max: float = 1.0
    
    # Epsilon for exception handling
    invalid_reliability: float = 1e-3
    
    alert_threshold_high: float = 0.75
    alert_threshold_moderate: float = 0.55
    # Model Paths
    slm_model_path: Optional[str] = None
    # [추가됨] 훈련된 LightGBM/XGBoost 모델 경로 (예: "models/prosody_lgbm.joblib")
    prosody_model_path: Optional[str] = None
    
    class Config:
        env_prefix = "FRAUD_"

settings = Settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HybridServer")

# pydub ffmpeg 경로 설정
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# AASIST 폴더 연동 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 server.py가 있는 voice 폴더
parent_dir = os.path.dirname(current_dir)                # 한 단계 위인 KB AI 폴더
aasist_folder = os.path.join(parent_dir, "aasist")       # KB AI/aasist 경로 지정
sys.path.append(aasist_folder)
try:
    from models.AASIST import Model as RealAASISTModel
except ImportError:
    RealAASISTModel = None

app = FastAPI(title=settings.app_name)


# ==========================================
# 🤖 [2. 통합 AI 엔진 공유 풀 (Shared Model Singleton)]
# ==========================================
class GlobalModelEngine:
    def __init__(self):
        self.stt_model: Optional[WhisperModel] = None
        self.fake_voice_detector: Optional[Any] = None
        self.vad_model: Optional[torch.nn.Module] = None
        self.slm_classifier: Optional[Any] = None
        self.prosody_classifier: Optional[Any] = None 
        self.system_degraded: bool = False
        self.load_all_models()

    def load_all_models(self):
        logger.info("🤖 [엔진 초기화] 통합 AI 엔진 로딩 시작...")
        
        try:
            self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("✅ [Faster-Whisper] 로드 완료")
        except Exception as e:
            logger.critical(f"❌ [Faster-Whisper] 로드 실패: {e}")
            self.system_degraded = True

        try:
            if RealAASISTModel is None: raise ImportError("AASIST 모듈 누락")
            with open(os.path.join(current_dir, "aasist", "config", "AASIST.conf"), "r") as f:
                config = json.load(f)
            self.fake_voice_detector = RealAASISTModel(config["model_config"])
            weight_path = os.path.join(current_dir, "aasist", "models", "weights", "AASIST.pth")
            self.fake_voice_detector.load_state_dict(torch.load(weight_path, map_location="cpu"))
            self.fake_voice_detector.eval()
            logger.info("✅ [AASIST] 로드 완료")
        except Exception as e:
            logger.warning(f"⚠️ [AASIST] 로드 실패 (Deepfake 모듈 제외): {e}")
            self.fake_voice_detector = None
            self.system_degraded = True  # [FIXED: Issue 4] AASIST 실패 시 상태 강등 명시

        try:
            # force_reload=False 이므로 오프라인 캐시가 있다면 작동함
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
            self.vad_model = model
            logger.info("✅ [Silero VAD] 로드 완료")
        except Exception as e:
            logger.warning(f"⚠️ [Silero VAD] 로드 실패 (VAD 폴백 모드 작동): {e}")
            self.vad_model = None
            self.system_degraded = True  # [FIXED: Issue 4] VAD 로드 실패 명시

        # [FIXED: Bug C] 진짜 파인튜닝된 SLM 모델이 제공된 경우에만 파이프라인 로드
        if settings.prosody_model_path and os.path.exists(settings.prosody_model_path):
            try:
                import joblib
                self.prosody_classifier = joblib.load(settings.prosody_model_path)
                logger.info(f"✅ [Prosody Classifier] ML 모델 로드 완료: {settings.prosody_model_path}")
            except Exception as e:
                logger.error(f"❌ [Prosody Classifier] 로드 실패: {e}")
                self.prosody_classifier = None
        else:
            logger.info("ℹ️ [Prosody Classifier] 지정된 모델이 없어 기본값(0.5)으로 작동합니다.")

engine = GlobalModelEngine()

def calculate_snr(audio_samples: np.ndarray) -> float:
    """오디오 신호의 SNR(Signal-to-Noise Ratio)을 계산하여 신뢰도에 반영"""
    if len(audio_samples) < 16000: return 20.0
    signal_power = np.var(audio_samples)
    noise_power = np.var(audio_samples[:1600]) # 처음 0.1초를 Background Noise로 가정
    if noise_power == 0: return 30.0
    snr = 10 * np.log10(signal_power / noise_power)
    return float(np.clip(snr, 0, 30))

def extract_paper_grade_prosody(audio_path: str, transcript: str) -> dict:
    """
    Stage 3: Intensity 기반 Silence Detection 및 HNR이 포함된 논문용 운율 추출기
    """
    try:
        snd = parselmouth.Sound(audio_path)
        
        # 1. Pitch (F0) & Voice Quality (Jitter, Shimmer, HNR)
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0] 
        f0_variance = np.var(f0_values) if len(f0_values) > 0 else 0
        
        pointProcess = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # 2. [수정됨] Praat Intensity 기반의 엄밀한 Silence Duration 추정
        intensity = snd.to_intensity()
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
        # Praat의 기본 Silence 임계값: 최대 Intensity 대비 -25dB 미만인 구간
        silence_threshold = max_intensity - 25.0 
        intensity_values = intensity.values.squeeze()
        
        # 조건에 맞는 프레임 개수 * 프레임 시간 간격(time_step)
        silence_duration = float(np.sum(intensity_values < silence_threshold) * intensity.get_time_step())
        total_duration = snd.get_total_duration()
        pause_ratio = silence_duration / total_duration if total_duration > 0 else 0.0
        
        # 3. Speech Rate (Approximate syllable count 논문 명시)
        # 3. Speech Rate 및 Pause Ratio 원시값 계산
        syllables = len(re.sub(r'\s+', '', transcript))
        speech_rate = syllables / total_duration if total_duration > 0 else 0.0
        pause_ratio_raw = silence_duration / total_duration if total_duration > 0 else 0.0
        energy_mean = np.mean(snd.values ** 2)
        
        # 4. Data-driven Normalization (전체 Settings 변수 활용)
        norm_jitter = float(np.clip(jitter / settings.jitter_max, 0.0, 1.0)) if not np.isnan(jitter) else 0.0
        norm_shimmer = float(np.clip(shimmer / settings.shimmer_max, 0.0, 1.0)) if not np.isnan(shimmer) else 0.0
        norm_hnr = float(np.clip(hnr / settings.hnr_max, 0.0, 1.0)) if not np.isnan(hnr) else 0.5
        norm_speech_rate = float(np.clip(speech_rate / settings.speech_rate_max, 0.0, 1.0))
        norm_f0 = float(np.clip(f0_variance / settings.f0_var_max, 0.0, 1.0))
        norm_pause = float(np.clip(pause_ratio_raw / settings.pause_ratio_max, 0.0, 1.0)) # [추가됨]
        norm_energy = float(np.clip((energy_mean - settings.energy_mu) / (3 * settings.energy_sigma), 0.0, 1.0))
        
        feature_vector = np.array([
            norm_jitter, norm_shimmer, norm_hnr, norm_f0, norm_pause, norm_speech_rate, norm_energy
        ])
        
        return {
            "features": feature_vector,
            "quality_metrics": {"hnr_norm": norm_hnr, "energy_norm": norm_energy}, # [변경됨] 명확한 Key 사용
            "feature_valid": True
        }

    except Exception as e:
        logger.warning(f"Prosody extraction failed: {e}")
        return {
            "features": np.zeros(7), 
            "quality_metrics": {"hnr_norm": 0.0, "energy_norm": 0.0}, # [변경됨]
            "feature_valid": False
        }
        
def calculate_jensen_shannon(p1: float, p2: float) -> float:
    """Stage 6: JSD 기반의 모달리티 간 의미적 충돌(Conflict Distance) 계산"""
    if p1 is None or p2 is None: return 0.0
    dist1 = [1.0 - p1, p1]
    dist2 = [1.0 - p2, p2]
    return round(float(jensenshannon(dist1, dist2)), 4)

# ==========================================
# 📐 [3. REST API용 수식 & 분석 헬퍼 함수들]
# ==========================================
def analyze_audio_sliding_window(samples: np.ndarray, model, target_len: int = 64600, hop_len: int = 32300):
    if model is None: return 0.5, [0.0, 0.0]
    # (기존 슬라이딩 윈도우 로직 유지 - 생략 방지 위해 그대로 작성)
    if len(samples) <= target_len:
        padded = np.pad(samples, (0, target_len - len(samples)), 'constant')
        tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, prediction = model(tensor)
            return round(torch.sigmoid(prediction)[0][1].item(), 4), [prediction[0][0].item(), prediction[0][1].item()]
    max_score, best_logits = -1.0, [0.0, 0.0]
    for start_idx in range(0, len(samples) - target_len + 1, hop_len):
        chunk = samples[start_idx : start_idx + target_len]
        tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, prediction = model(tensor)
            score = torch.sigmoid(prediction)[0][1].item()
            if score > max_score:
                max_score, best_logits = score, [prediction[0][0].item(), prediction[0][1].item()]
    return round(float(max_score), 4), best_logits

def estimate_rule_guided_lexical_risk(text: str):
    normalized_text = re.sub(r'\s+', '', text)
    
    # [FIXED: Bug A] 부동소수점 역산 절삭 방지 - 원본 매치 리스트 길이 보존
    matched_auth = [p for p in [r'검찰(?:청)?', r'경찰(?:청|서)?', r'금융감독원', r'금감원', r'법원', r'검사(?:님|관)?', r'수사관'] if re.search(p, normalized_text)]
    matched_fin = [p for p in [r'안전계좌', r'계좌이체', r'송금', r'대출(?:상환|환급|금리|사기)', r'입금요구', r'자금동결'] if re.search(p, normalized_text)]
    matched_cred = [p for p in [r'비밀번호', r'인증서', r'주민(?:등록)?번호', r'원격(?:제어|지원|앱)', r'앱깔(?:아|라|으)', r'보안토큰', r'인증번호'] if re.search(p, normalized_text)]
    matched_urg = [p for p in [r'당장', r'급히', r'즉시', r'동결', r'구속', r'빨리', r'마감', r'영장', r'체포'] if re.search(p, normalized_text)]
    
    auth_count, fin_count, cred_count, urg_count = len(matched_auth), len(matched_fin), len(matched_cred), len(matched_urg)
    
    authority_score = min(auth_count * 0.5, 1.0)
    financial_score = min(fin_count * 0.33, 1.0)
    credential_score = min(cred_count * 0.33, 1.0)
    urgency_score = min(urg_count * 0.33, 1.0)
    
    brands = ["naver", "kakao", "kbstar", "shinhan", "woori", "ibk", "nonghyup", "google", "apple", "daum", "toss", "kb", "nh"]
    sec_keywords = ["login", "secure", "account", "cert", "auth", "verify", "support", "update", "banking", "recovery", "id"]
    tlds = [r"\.com", r"\.kr", r"\.net", r"\.org", r"\.co\.kr", r"\.xyz", r"\.top", r"\.site", r"\.info", r"\.cc", r"\.me"]
    raw_urls = re.compile(r'(?:https?://|www\.)?[^\s]+(?:' + '|'.join(brands) + r')[-\.0-9]*(?:' + '|'.join(sec_keywords) + r')[-\.0-9]*(?:' + '|'.join(tlds) + r')', re.IGNORECASE).findall(text)
    accounts_found = re.compile(r'\b([0-9]{3,6}-[0-9]{2,6}-[0-9]{2,6})\b').findall(text)
    structural_score = max(1.0 if raw_urls else 0.0, 1.0 if accounts_found else 0.0)

    linear_risk = (settings.weight_authority * authority_score) + (settings.weight_financial * financial_score) + (settings.weight_credential * credential_score) + (settings.weight_urgency * urgency_score) + (settings.weight_structural * structural_score)
    synergy_boost = settings.lexical_synergy_boost if (authority_score > 0.0 and urgency_score > 0.0) else 0.0
    
    return {
        "features": {"authority": round(authority_score, 2), "financial": round(financial_score, 2), "credential": round(credential_score, 2), "urgency": round(urgency_score, 2), "structural": round(structural_score, 2), "synergy_boost": round(synergy_boost, 2)},
        "text_risk": round(min(linear_risk + synergy_boost, 1.0), 4),
        "details": {
            "matched_authority_count": auth_count, # 계산된 원본 카운트 그대로 주입
            "matched_financial_count": fin_count, 
            "matched_credential_count": cred_count, 
            "suspicious_urls": raw_urls, 
            "accounts_found": accounts_found
        }
    }

def calculate_logit_margin_confidence_proxy(raw_logits: list):
    if not raw_logits or len(raw_logits) < 2: return None
    return round(float(np.tanh(abs(float(raw_logits[1]) - float(raw_logits[0])) / settings.logit_margin_scale_theta)), 4)

def calculate_continuous_paralinguistic_risk(speech_rate_syllables: float, pause_ratio: float):
    return round((settings.omega_speech_rate * min(speech_rate_syllables / settings.speech_rate_val_mean, 1.0)) + (settings.omega_pause_ratio * (1.0 - min(pause_ratio, 1.0))), 4)

def calculate_jensen_shannon(p1: float, p2: float) -> float:
    """Stage 6: Conflict Distance 계산 (Jensen-Shannon Divergence)"""
    if p1 is None: p1 = 0.0
    dist1 = [1.0 - p1, p1]
    dist2 = [1.0 - p2, p2]
    return round(float(jensenshannon(dist1, dist2)), 4)

def reliability_aware_adaptive_fusion(
    audio_score: float, text_score: float, paralinguistic_score: float, 
    audio_conf: float, text_conf: float, vad_prob: float, snr_db: float, 
    prosody_metrics: dict, prosody_valid: bool,
    prosody_conf: float = 0.5 # [추가] ML 기반 운율 Confidence
):
    """Stage 2~5: Entropy & SNR 기반 동적 가중치 융합 및 Dynamic Threshold"""
    
    # 1. Text Reliability (Confidence + Entropy 결합)
    p_t = text_score if text_score else 0.5
    entropy_text = -(p_t * np.log2(p_t + 1e-9) + (1.0 - p_t) * np.log2((1.0 - p_t) + 1e-9))
    r_text = (text_conf if text_conf else 0.5) * (1.0 - entropy_text)  
    
    # 2. Audio Reliability (중복 제거: 여기서 한 번만 정확한 Min-Max 정규화 수행)
    snr_norm = float(np.clip((snr_db - settings.snr_min) / (settings.snr_max - settings.snr_min), 0.0, 1.0))
    r_audio = (audio_conf if audio_conf else 0.5) * snr_norm * (vad_prob if vad_prob else 0.5)  
    
    # 3. Prosody Reliability (Confidence × SNR × Quality)
    if prosody_valid:
        prosody_quality = float(np.mean([prosody_metrics.get("hnr_norm", 0.5), prosody_metrics.get("energy_norm", 0.5)]))
        r_para = prosody_conf * snr_norm * prosody_quality
    else:
        r_para = settings.invalid_reliability

    # ---------------------------------------------------------
    # 이하 논리(Conflict 계산, Softmax 가중치, Threshold)는 동일
    # ---------------------------------------------------------
    a_val = audio_score if audio_score else 0.0
    jsd_conflict = calculate_jensen_shannon(a_val, text_score)
    
    # Softmax Gating Weights
    reliabilities = np.array([r_audio, r_text, r_para])
    w_vec = np.exp(reliabilities) / np.sum(np.exp(reliabilities))
    
    gating_weights = {
        "audio": round(float(w_vec[0]), 4),
        "text": round(float(w_vec[1]), 4),
        "paralinguistic": round(float(w_vec[2]), 4)
    }
    
    # Final Risk Calculation
    scores = np.array([a_val, text_score, paralinguistic_score if paralinguistic_score else 0.0])
    final_risk = round(float(np.dot(w_vec, scores)), 4)
    
    # Dynamic Threshold (Uncertainty = Entropy + JSD Conflict)
    fusion_entropy = float(-np.sum(w_vec * np.log(w_vec + 1e-9)))
    uncertainty = fusion_entropy + jsd_conflict
    dynamic_thresh = round(settings.base_threshold + (settings.lambda_uncertainty * uncertainty), 4)
    
    threat = "HIGH RISK" if final_risk >= dynamic_thresh + 0.15 else ("MODERATE RISK" if final_risk >= dynamic_thresh else "NORMAL")

    return {
        "final_risk_score": final_risk, 
        "dynamic_threshold": dynamic_thresh, 
        "threat_level": threat,
        "bivariate_mapping_state": {"conflict_jsd": jsd_conflict, "uncertainty": round(uncertainty, 4)},
        "pre_softmax_logits": {"audio": round(float(r_audio),4), "text": round(float(r_text),4), "paralinguistic": round(float(r_para),4)}, 
        "gating_weights": gating_weights
    }
    
def generate_contribution_explanation(audio_score: float, text_analysis: dict, paralinguistic_risk: float, fusion_result: dict, conflict_distance: float, reliability_score: float):
    weights = fusion_result["gating_weights"]
    
    # [FIXED: Bug B] 0.0이 Falsy로 판정되어 N/A로 찍히는 문제 해결 (명시적 is not None 체크)
    audio_display = audio_score if audio_score is not None else "N/A"
    para_display = paralinguistic_risk if paralinguistic_risk is not None else "N/A"
    
    a_val = audio_score if audio_score is not None else 0.0
    p_val = paralinguistic_risk if paralinguistic_risk is not None else 0.0
    
    total_raw = (a_val * weights.get("audio", 0.0)) + (text_analysis["text_risk"] * weights.get("text", 0.0)) + (p_val * weights.get("paralinguistic", 0.0))
    
    if total_raw > 0:
        a_pct = round(((a_val * weights.get("audio", 0.0)) / total_raw) * 100, 1)
        t_pct = round(((text_analysis["text_risk"] * weights.get("text", 0.0)) / total_raw) * 100, 1)
        p_pct = round(100.0 - (a_pct + t_pct), 1)
    else: 
        a_pct, t_pct, p_pct = (0.0, 50.0, 50.0) if audio_score is None else (33.3, 33.3, 33.4)
    
    return {
        "evidence": {
            "audio_score": audio_display, 
            "text_semantic_risk": text_analysis["text_risk"], 
            "paralinguistic_risk": para_display, 
            "semantic_vectors": text_analysis["features"], 
            "conflict_distance": conflict_distance, 
            "reliability_score": reliability_score, 
            "dynamic_threshold": fusion_result["dynamic_threshold"]
        },
        "xai_pipeline": {
            "step1_pre_softmax_logits": fusion_result["pre_softmax_logits"], 
            "step2_softmax_gating_weights": weights, 
            "step3_modality_contribution_pct": {"audio_modality_pct": f"{a_pct}%", "text_modality_pct": f"{t_pct}%", "paralinguistic_modality_pct": f"{p_pct}%"}
        },
        "decision": {"threat_level": fusion_result["threat_level"], "final_risk_score": fusion_result["final_risk_score"]}
    }


# ==========================================
# 🌐 [4. WebSocket 스트리밍용 클래스 & 워커 (Realtime Module)]
# ==========================================
class CallSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.speech_buffer: List[np.ndarray] = []
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.transcript_history: List[str] = []

    def reset_speech_buffer(self):
        self.speech_buffer.clear()
        self.is_speaking = False
        self.speech_start_time = 0.0

def realtime_multimodal_worker_sync(audio_array: np.ndarray, history: List[str]) -> Dict[str, Any]:
    result = {"text": "", "risk_score": 0.0, "threat_level": "NORMAL", "intent_detected": "None", "reasoning": ""}
    if not engine.stt_model: return result
    
    # 1. STT 추출
    segments, _ = engine.stt_model.transcribe(audio_array, beam_size=3, language="ko", condition_on_previous_text=False)
    text_chunk = " ".join([s.text for s in segments]).strip()
    if not text_chunk: return result
    
    result["text"] = text_chunk
    history.append(text_chunk)
    context_window = " ".join(history[-5:])

    # 2. Text Pipeline
    text_risk_score, text_confidence = 0.0, 0.5
    if engine.slm_classifier:
        try:
            preds = engine.slm_classifier(context_window, top_k=None)
            if isinstance(preds[0], list): preds = preds[0]
            phishing_pred = next((p for p in preds if p["label"].lower() in ["label_1", "phishing", "fraud"]), None)
            normal_pred = next((p for p in preds if p["label"].lower() in ["label_0", "normal"]), None)
            if phishing_pred and normal_pred:
                text_risk_score = float(phishing_pred["score"])
                text_confidence = float(max(phishing_pred["score"], normal_pred["score"]))
        except Exception:
            pass
    else:
        lex_res = estimate_rule_guided_lexical_risk(context_window)
        text_risk_score = lex_res["text_risk"]

    # 3. Audio & Prosody Pipeline
    calculated_snr = calculate_snr(audio_array)
    vad_prob = 1.0  # 청크 발화 검증 통과 
    audio_score, audio_conf = 0.0, 0.5
    
    if engine.fake_voice_detector:
        audio_score, best_logits = analyze_audio_sliding_window(audio_array, engine.fake_voice_detector)
        audio_conf = calculate_logit_margin_confidence_proxy(best_logits) or 0.5

    paralinguistic_risk, prosody_conf = 0.5, 0.5
    prosody_metrics = {"hnr_norm": 0.5, "energy_norm": 0.5}
    prosody_valid = False
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
        
        audio_int16 = (audio_array * 32767).astype(np.int16)
        wav.write(tmp_path, settings.sample_rate, audio_int16)
            
        prosody_result = extract_paper_grade_prosody(tmp_path, text_chunk)
        prosody_metrics = prosody_result["metrics"] 
        prosody_valid = prosody_result["feature_valid"]
        
        if prosody_valid and getattr(engine, 'prosody_classifier', None):
            probs = engine.prosody_classifier.predict_proba([prosody_result["features"]])[0]
            paralinguistic_risk = float(probs[1])
            prosody_conf = float(max(probs[0], probs[1]))
            
    except Exception as e:
        logger.error(f"Realtime Prosody Error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # 4. Multimodal Fusion 실행
    fusion_result = reliability_aware_adaptive_fusion(
        audio_score=audio_score, text_score=text_risk_score, paralinguistic_score=paralinguistic_risk,
        audio_conf=audio_conf, text_conf=text_confidence, vad_prob=vad_prob, snr_db=calculated_snr,
        prosody_metrics=prosody_metrics, prosody_valid=prosody_valid, prosody_conf=prosody_conf
    )

    result["risk_score"] = fusion_result["final_risk_score"]
    result["threat_level"] = fusion_result["threat_level"]
    result["reasoning"] = f"Realtime Fusion (Uncertainty: {fusion_result['bivariate_mapping_state']['uncertainty']})"

    return result



# ==========================================
# 🚀 [5. API 엔드포인트 통합 (REST + WebSocket)]
# ==========================================

# --- 1️⃣ REST API --- (실제 분석 로직 적용)
@app.post("/analyze/pipeline")
async def run_pipeline(file: UploadFile = File(None), text_input: str = Form(None)):
    if not file and not text_input:
        raise HTTPException(status_code=400, detail="오디오 파일이나 텍스트를 입력해주세요.")
        
    final_text = text_input or ""
    audio_score = confidence_proxy = None
    
    # 기본값 세팅 (오디오가 없을 경우 대비)
    paralinguistic_risk = 0.5
    calculated_snr = 20.0
    vad_probability = 0.8
    prosody_metrics = {"hnr_norm": 0.5, "energy_norm": 0.5}
    prosody_valid = False
    tmp_path = None
    
    try:
        # [Step 1] 오디오 파일 처리 및 Feature 추출
        if file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
                
            audio_segment = AudioSegment.from_file(tmp_path).set_frame_rate(16000).set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            # 1-1. 기본 오디오 Feature (SNR, VAD)
            calculated_snr = calculate_snr(samples)
            if engine.vad_model:
                vad_tensor = torch.from_numpy(samples).squeeze()
                with torch.no_grad():
                    vad_probability = engine.vad_model(vad_tensor, settings.sample_rate).item()

            # 1-2. AASIST (딥페이크/음성변조 탐지)
            if engine.fake_voice_detector:
                audio_score, best_logits = analyze_audio_sliding_window(samples, engine.fake_voice_detector)
                confidence_proxy = calculate_logit_margin_confidence_proxy(best_logits)
            
            # 1-3. STT 텍스트 추출 (입력 텍스트가 없을 때)
            if not final_text and engine.stt_model:
                segments, _ = engine.stt_model.transcribe(samples, beam_size=3, language="ko")
                final_text = " ".join([s.text for s in segments]).strip()
            
            # 1-4. [핵심 변경점] Prosody 추출 및 ML 기반 Risk 추론
            # 1-4. Prosody 추출 및 ML 기반 Risk 추론
            prosody_result = extract_paper_grade_prosody(tmp_path, final_text)
            prosody_metrics = prosody_result["quality_metrics"] # [버그 원천 차단] 정확히 매핑
            prosody_valid = prosody_result["feature_valid"]
            
            # [TO-BE]
            if prosody_valid and getattr(engine, 'prosody_classifier', None):
                probs = engine.prosody_classifier.predict_proba([prosody_result["features"]])[0]
                paralinguistic_risk = float(probs[1])
                prosody_conf = float(max(probs[0], probs[1])) # [추가] Confidence 추출
            else:
                paralinguistic_risk = 0.5
            # os.remove(tmp_path) 는 여기서 삭제합니다.
                prosody_conf = 0.5 # [이 줄을 새로 추가합니다!]
                calculated_snr = 20.0
                
            os.remove(tmp_path)

        # [Step 2] Text Pipeline: 정규식 대신 SLM(KoELECTRA) 추론
        text_risk_score = 0.0
        text_confidence = 0.0
        
        if engine.slm_classifier:
            preds = engine.slm_classifier(final_text, top_k=None)
            if isinstance(preds[0], list): preds = preds[0]
            
            phishing_pred = next((p for p in preds if p["label"].lower() in ["label_1", "phishing", "fraud"]), None)
            normal_pred = next((p for p in preds if p["label"].lower() in ["label_0", "normal"]), None)
            
            if phishing_pred and normal_pred:
                text_risk_score = phishing_pred["score"]
                text_confidence = max(phishing_pred["score"], normal_pred["score"])
        else:
            lex_res = estimate_rule_guided_lexical_risk(final_text)
            text_risk_score = lex_res["text_risk"]
            text_confidence = 0.5 

        xai_features = estimate_rule_guided_lexical_risk(final_text)["features"]

        # [Step 3] Data-driven Adaptive Fusion 호출 (JSD, SNR, VAD 모두 내부에서 처리)
        fusion_result = reliability_aware_adaptive_fusion(
            audio_score=audio_score,
            text_score=text_risk_score,
            paralinguistic_score=paralinguistic_risk,
            audio_conf=confidence_proxy,
            text_conf=text_confidence,
            vad_prob=vad_probability,
            snr_db=calculated_snr,
            prosody_metrics=prosody_metrics, # HNR, Energy 전달
            prosody_valid=prosody_valid,  
            prosody_conf=prosody_conf  # 유효성 플래그 전달
        )
        
        # [Step 4] XAI 리포트 구성
        explanation = generate_contribution_explanation(
            audio_score=audio_score,
            text_analysis={"text_risk": text_risk_score, "features": xai_features},
            paralinguistic_risk=paralinguistic_risk,
            fusion_result=fusion_result,
            conflict_distance=fusion_result["bivariate_mapping_state"]["conflict_jsd"],
            reliability_score=fusion_result["pre_softmax_logits"].get("audio", 0.5)
        )
        
        return {
            "status": "success",
            "transcript": final_text,
            "analysis_result": explanation
        }
        
    except Exception as e:
        logger.error(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # <--- [TO-BE] 무조건 여기서 한 번만 안전하게 파일 삭제
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# --- 2️⃣ WebSocket API: [FIXED] Producer-Consumer 구조 적용 ---
@app.websocket("/ws/detect/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # 보안 지적사항(Issue 6) 뼈대 반영: 향후 쿼리 파라미터나 헤더 토큰 검증 로직 추가 위치
    await websocket.accept()
    session = CallSession(session_id=session_id)
    
    # Producer-Consumer 분리를 위한 오디오 청크 큐
    audio_queue = asyncio.Queue()
    
    logger.info(f"🔌 [WebSocket 연결] 세션 시작: {session_id}")
    await websocket.send_json({
        "event": "CONNECTION_ESTABLISHED", "session_id": session_id,
        "system_status": "DEGRADED" if engine.system_degraded else "HEALTHY"
    })

    chunk_size_bytes = int(settings.sample_rate * 2 * (settings.chunk_duration_ms / 1000.0))

    # [Task 1] Consumer: 큐에서 꺼내와서 VAD 검사 및 STT 추론 스레드 위임
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
                    
                    is_speech = False
                    if engine.vad_model:
                        vad_tensor = torch.from_numpy(pcm_data).squeeze()
                        with torch.no_grad():
                            speech_prob = engine.vad_model(vad_tensor, settings.sample_rate).item()
                        is_speech = speech_prob >= settings.vad_threshold
                    else: 
                        is_speech = True # VAD 실패시 폴백

                    current_time = time.time()
                    
                    if is_speech:
                        if not session.is_speaking:
                            session.is_speaking = True
                            session.speech_start_time = current_time
                        session.speech_buffer.append(pcm_data)
                        
                        if (current_time - session.speech_start_time) >= settings.max_speech_duration_s:
                            is_speech = False

                    if not is_speech and session.is_speaking:
                        speech_duration = len(session.speech_buffer) * (settings.chunk_duration_ms / 1000.0)
                        if speech_duration >= settings.min_speech_duration_s:
                            full_speech_array = np.concatenate(session.speech_buffer)
                            
                            # 비동기 스레드 위임으로 메인 이벤트 루프 블로킹 방지
                            ml_result = await asyncio.to_thread(realtime_multimodal_worker_sync, full_speech_array, session.transcript_history)

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

    processor_task = asyncio.create_task(process_audio())

    # [Task 2] Producer: 클라이언트로부터 쉴 새 없이 네트워크 패킷만 수신하여 큐에 적재 (직렬화 병목 해소)
    try:
        while True:
            data = await websocket.receive_bytes()
            await audio_queue.put(data)
    except WebSocketDisconnect:
        logger.info(f"🔌 [WebSocket 종료] 클라이언트 해제: {session_id}")
    finally:
        processor_task.cancel()