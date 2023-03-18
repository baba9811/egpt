import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

# 데이터셋 불러오기
data = pd.read_csv('data.csv', encoding='cp949')

# KoGPT2 토크나이저 및 모델 불러오기
tokenizer = GPT2TokenizerFast.from_pretrained("skt/kogpt2-base-v2")
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("skt/kogpt2-base-v2")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", config=config)

# 데이터셋 전처리
def prepare_dataset(data):
    tokenized_examples = []
    for _, example in data.iterrows():
        context = str(example["context"])
        response = str(example["response"])
        input_ids = tokenizer(context, return_tensors="pt", padding='max_length', truncation=True, max_length=128).input_ids[0].tolist()
        output_ids = tokenizer(response, return_tensors="pt", padding='max_length', truncation=True, max_length=128).input_ids[0].tolist()
        # 예외 처리
        if not input_ids:
            continue
        if not output_ids:
            continue
        tokenized_examples.append({"input_ids": input_ids, "output_ids": output_ids})
    return tokenized_examples

# 학습 데이터셋 준비
tokenized_examples = prepare_dataset(data)
train_dataset = Dataset.from_dict({k: [dic[k] for dic in tokenized_examples] for k in tokenized_examples[0]}) if tokenized_examples else None
if train_dataset is not None:
    train_dataset = train_dataset.map(lambda x: {"input_ids": x["input_ids"][:127], "labels": x["output_ids"][:127]}, remove_columns=["output_ids"])
    train_dataset.set_format(type="torch", columns=["input_ids", "labels"])


# 학습 설정
training_args = TrainingArguments(
    output_dir="./kogpt2_fine_tuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# 학습
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        pad_to_multiple_of=128,
        mlm=False,
    ),
    train_dataset=train_dataset,
)

trainer.train()

# 토크나이저와 모델 저장
tokenizer.save_pretrained("./kogpt2_fine_tuned")
model.save_pretrained("./kogpt2_fine_tuned")
