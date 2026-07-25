import os
import numpy as np
import imageio_ffmpeg
from pydub import AudioSegment
import torch
import torch.nn as nn

# 1. pydub에게 ffmpeg 위치 알려주기
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# --- ✂️ 소리 다듬는 가위 ---
def clean_my_voice(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    
    sound = AudioSegment.from_file(file_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    
    samples = np.array(sound.get_array_of_samples(), dtype=np.float32) / 32768.0
    print(f"✂️ 가위질 완료! (길이: {len(samples)/16000:.2f}초)")
    return samples

# --- 🧠 미니 AI 뇌 (탐정 로봇) ---
class MiniBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.magnifying_glass = nn.Linear(1, 1)
        self.score_maker = nn.Sigmoid() 

    def forward(self, sound_box):
        compressed_sound = sound_box.mean(dim=1, keepdim=True) 
        x = self.magnifying_glass(compressed_sound)
        final_score = self.score_maker(x)
        return final_score


# ==========================================
# 🚀 여기서부터 진짜 실행되는 부분입니다! 🚀
# ==========================================

# 1. 소리 다듬기
audio_signal = clean_my_voice("test_voice.wav")

# 2. 방 밖에서 포장 상자(텐서) 만들기! (이제 에러 안 남!)
audio_tensor = torch.tensor(audio_signal, dtype=torch.float32).unsqueeze(0)
print(f"📦 뇌로 들어갈 상자 모양(Shape): {audio_tensor.shape}")

# 3. 미니 뇌 깨우기
print("\n🤖 미니 AI 뇌를 조립했습니다!")
my_robot_brain = MiniBrain()

# 4. 상자를 뇌에 집어넣고 진짜 점수 뽑아내기!
print("⚡ 소리 상자를 뇌에 통과시킵니다... 찌릿찌릿!")
real_calculated_score = my_robot_brain(audio_tensor)

print(f"🎯 AI가 직접 계산한 가짜 확률 점수: {real_calculated_score.item():.4f}")

# ==========================================
# 🧠 뇌 이식 수술 연습하기 (.pth 파일 다루기)
# ==========================================

import os

# "지금 이 파이썬 코드가 있는 정확한 폴더 주소"를 알아내는 마법 주문이야!
current_folder = os.path.dirname(os.path.abspath(__file__))
save_address = os.path.join(current_folder, "my_brain_memory.pth")

# 1. 뇌에서 기억 구슬 뽑아내기 (정확한 주소에 저장!)
print("\n💾 미니 뇌의 기억을 지정된 폴더에 저장합니다...")
torch.save(my_robot_brain.state_dict(), save_address)
print(f"✅ 구슬 생성 완료! 위치: {save_address}")

# 2. 공장에서 갓 나온 새로운 '빈 뇌' 준비하기
print("\n🔄 아무것도 모르는 텅 빈 새로운 뇌를 가져옵니다...")
new_robot_brain = MiniBrain()

# 3. 새로운 뇌에 아까 뽑아둔 기억 구슬 쏙 끼워넣기! (이식 수술)
print("💉 빈 뇌에 똑똑한 기억 구슬(.pth)을 이식합니다... 찌릿찌릿!")
new_robot_brain.load_state_dict(torch.load(save_address, weights_only=True))
print("🎯 수술 성공! 새로운 뇌도 완벽하게 똑똑해졌어!")

# 4. 수술받은 새 뇌가 점수를 잘 내는지 테스트!
new_score = new_robot_brain(audio_tensor)
print(f"✨ 이식받은 새 뇌의 가짜 확률 점수: {new_score.item():.4f}")