from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 저장된 모델 및 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained("chatbot_model")
model = GPT2LMHeadModel.from_pretrained("chatbot_model")

# 챗봇 입력에 대한 답변 생성
def generate_response(input_text):
    input_tokenized = tokenizer.encode_plus("[CLS]" + input_text + "[SEP]", max_length=1024, pad_to_max_length=True, return_tensors='pt')
    input_ids = input_tokenized['input_ids']
    attention_mask = input_tokenized['attention_mask']
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=1024, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 챗봇 실행
while True:
    input_text = input("사용자 입력: ")
    if input_text.strip().lower() == "exit":
        break
    response = generate_response(input_text)
    print("챗봇 응답:", response)
