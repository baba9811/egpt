import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(model, tokenizer, context, max_length=50):
    input_ids = tokenizer.encode(context, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('skt/kogpt2-base-v2')
    model = GPT2LMHeadModel.from_pretrained('./results/checkpoint-last')

    print("대화를 시작합니다. 'exit'를 입력하면 대화를 종료합니다.")
    while True:
        context = input('User: ')
        if context.lower() == 'exit':
            break
        response = generate_response(model, tokenizer, context)
        print(f'Chatbot: {response}')

if __name__ == '__main__':
    main()
