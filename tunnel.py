from pyngrok import ngrok
import time

# 🔑 여기에 방금 홈페이지에서 복사한 긴 토큰을 붙여넣어 줘! (따옴표는 지우지 마!)
ngrok.set_auth_token("3GqnFNU9D7aJSTS9txuDw0LybcW_5rjpwKeB4Sroh7AuyCfwz")

print("🌍 우체통과 인터넷을 연결하는 마법 터널을 뚫는 중...")

# 8000번 포트(우리가 켠 FastAPI 서버)로 터널 연결!
tunnel = ngrok.connect(8000)

print("\n" + "="*50)
print("🎉 터널 개방 성공! 팀원에게 아래 주소를 알려주세요!")
print(f"👉 {tunnel.public_url}")
print("="*50 + "\n")
print("⚠️ 주의: 이 검은 창을 끄면 터널도 닫힙니다.")

while True:
    time.sleep(1)