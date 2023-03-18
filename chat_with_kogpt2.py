from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 모델 및 토크나이저 불러오기
tokenizer = GPT2TokenizerFast.from_pretrained("./kogpt2_fine_tuned")
model = GPT2LMHeadModel.from_pretrained("./kogpt2_fine_tuned")

# 대화 시작
print("챗봇과 대화를 시작합니다. 'exit'를 입력하면 종료됩니다.")
while True:
    input_text = input("나: ")
    if input_text.lower() == "exit":
        break

    # 사용자 입력 텍스트를 토큰화하고 모델에 전달
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_tokens, max_length=50, num_return_sequences=1)

    # 생성된 토큰을 텍스트로 변환
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"챗봇: {output_text}")
