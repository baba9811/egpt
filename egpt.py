import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd

# 데이터셋 불러오기
df = pd.read_csv('data.csv', encoding='cp949')

# KoGPT2 tokenizer와 모델 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 입력 문장 tokenizing
inputs = []

for i in df['context']:
    inputs.append(tokenizer.encode(str(i)))

# 출력 문장 tokenizing
outputs = []

for i in df['response']:
    outputs.append(tokenizer.encode("[BOS] " + str(i) + " [EOS]"))


# PyTorch Dataset 클래스 상속
class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs[index])
        output_ids = torch.tensor(self.outputs[index])
        return input_ids, output_ids

# DataLoader 정의
dataset = MyDataset(inputs, outputs)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Trainer와 TrainingArguments 정의
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataloader,
    data_collator=lambda data: {'input_ids': torch.stack([item[0] for item in data]),
                                'labels': torch.stack([item[1] for item in data])},
)


# 모델 학습
trainer.train()

# 파인튜닝한 모델과 tokenizer 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("results/checkpoint-5000")

while True:
    # 사용자 입력 받기
    user_input = input("User: ")

    # 'exit' 입력 시 프로그램 종료
    if user_input == 'exit':
        break

    # 모델 입력을 위해 tokenizer 사용
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # 모델에 입력하여 출력 생성
    output = model.generate(input_ids, max_length=100, do_sample=True)

    # 출력을 텍스트 형태로 변환하여 출력
    print("Chatbot:", tokenizer.decode(output[0], skip_special_tokens=True))
