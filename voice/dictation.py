import os
from faster_whisper import WhisperModel

# ==========================================
# 🤖 로봇 2호 + 3호 합체 기계!
# ==========================================

def catch_voice_phishing(file_path):
    print("\n[1단계] 🤖 받아쓰기 탐정이 귀를 쫑긋 세웁니다...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    # 한국어로 타자 치기!
    segments, info = model.transcribe(file_path, beam_size=5, language="ko")
    
    # 로봇이 친 타자들을 하나의 긴 문장으로 예쁘게 풀로 붙이기
    full_text = ""
    for segment in segments:
        full_text = full_text + segment.text + " "
        
    print(f"\n📝 탐정이 적어온 내용: {full_text}")
    
    print("\n[2단계] 🕵️ 단어 판독기가 나쁜 단어를 찾기 시작합니다...")
    bad_words = ["계좌이체", "비밀번호", "검찰", "경찰", "앱 깔아", "원격", "대출", "도용"]
    
    bad_count = 0
    for word in bad_words:
        if word in full_text:  # 방금 받아적은 그 내용 안에서 찾기!
            print(f"🚨 삐용삐용! 사기꾼 단어 발견: '{word}'")
            bad_count = bad_count + 1
            
    risk_score = bad_count * 0.33
    if risk_score > 1.0:
        risk_score = 1.0
        
    print(f"\n📊 최종 보이스피싱 위험도 점수: {risk_score:.2f} / 1.00")
    return risk_score

# --- 합체 기계 테스트 해보기 ---
current_folder = os.path.dirname(os.path.abspath(__file__))
exact_file_address = os.path.join(current_folder, "test_voice.wav")

# 기계에 파일을 쏙 집어넣기!
catch_voice_phishing(exact_file_address)