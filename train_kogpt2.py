import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

# 데이터셋 불러오기
data = pd.read_csv('data.csv', encoding='cp949')

# KoGPT2 토크나이저 및 모델 불러오기
tokenizer = GPT2TokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 데이터셋 전처리
def prepare_dataset(data):
    input_texts = []
    output_texts = []
    for i, row in data.iterrows():
        context = str(row['context'])
        response = str(row['response'])
        input_texts.append(context)
        output_texts.append(response)
    return input_texts, output_texts

# train.txt 파일 생성
train_input_texts, train_output_texts = prepare_dataset(data)
with open("train_input.txt", "w", encoding="utf-8") as f1, open("train_output.txt", "w", encoding="utf-8") as f2:
    for input_text, output_text in zip(train_input_texts, train_output_texts):
        f1.write(input_text + "\n")
        f2.write(output_text + "\n")

# 학습 데이터셋 준비
train_data = TextDataset(
    tokenizer=tokenizer,
    file_path="train_input.txt",
    block_size=128,
)
train_labels = TextDataset(
    tokenizer=tokenizer,
    file_path="train_output.txt",
    block_size=128,
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./kogpt2_fine_tuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 학습
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    ),
    train_dataset=train_data,
    train_labels=train_labels,
)

trainer.train()

# 토크나이저와 모델 저장
tokenizer.save_pretrained("./kogpt2_fine_tuned")
model.save_pretrained("./kogpt2_fine_tuned")
