import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 데이터셋 불러오기
data = pd.read_csv('data.csv', encoding='cp949')

# KoGPT2 토크나이저 및 모델 불러오기
tokenizer = GPT2TokenizerFast.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 데이터셋 전처리
def prepare_dataset(data):
    dialogues = []
    for i, row in data.iterrows():
        context = row['context']
        response = row['response']
        score = row['score']
        dialogue = f"{context}  {response}"
        dialogues.append(dialogue)
    return dialogues

# 학습 데이터셋 준비
train_data = TextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
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
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    ),
    train_dataset=train_data,
)

trainer.train()
