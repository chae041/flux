from transformers import AutoTokenizer, AutoModelForCausalLM

# FLUX 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("black-forest-labs/FLUX.1-dev")
model = AutoModelForCausalLM.from_pretrained("black-forest-labs/FLUX.1-dev")

# 테스트 문장
inputs = tokenizer("안녕하세요. 당신은 누구십니까?", return_tensors="pt")

# 텍스트 생성
outputs = model.generate(**inputs, max_length=100)

# 결과 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
