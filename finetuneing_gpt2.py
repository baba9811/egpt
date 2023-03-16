import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import os

# Load data from CSV file
df = pd.read_csv('data.csv', encoding='cp949')
os.mkdir('results')

# Preprocess data
inputs = []
labels = []
for index, row in df.iterrows():
    context = row['context']
    response = row['response']
    score = row['score']
    input_text = context + tokenizer.eos_token + response
    label_text = score
    inputs.append(input_text)
    labels.append(label_text)

# Tokenize inputs and labels
input_ids = [tokenizer.encode(text, return_tensors='pt') for text in inputs]
label_ids = [torch.tensor(int(score), dtype=torch.float) for score in labels]

# Define data collator
class DataCollator:
    def __call__(self, batch):
        input_ids = torch.cat([b for b in batch], dim=0)
        labels = torch.stack([b for b in label_ids], dim=0)
        return {"input_ids": input_ids, "labels": labels}

# Define training arguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=1e-4,
    adam_epsilon=1e-8,
    weight_decay=0.01,
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=input_ids,
    data_collator=DataCollator(),
)

# Start training
trainer.train()
