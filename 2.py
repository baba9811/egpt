import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 데이터셋 로드
data = pd.read_csv("data.csv", encoding='cp949')

# 토크나이저 및 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2", cache_dir="local_cache")

model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 데이터셋 가공
input_ids = []
attention_masks = []
labels = []
for idx, row in data.iterrows():
    context = row["context"]
    response = row["response"]
    input_text = "[CLS]" + context + "[SEP]" + response + "[SEP]"
    input_tokenized = tokenizer.encode_plus(input_text, max_length=1024, pad_to_max_length=True, return_tensors='pt')
    input_ids.append(input_tokenized['input_ids'])
    attention_masks.append(input_tokenized['attention_mask'])
    labels.append(torch.tensor(int(row["score"])))

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.stack(labels)

# 모델 학습
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    optimizer.zero_grad()

    outputs = model(input_ids, attention_mask=attention_masks, labels=input_ids)
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), input_ids.view(-1))

    loss.backward()
    optimizer.step()

# 파인튜닝된 모델 저장
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_model")
