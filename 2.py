import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 데이터셋 로드
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        for idx, row in data.iterrows():
            context = str(row["context"])
            response = str(row["response"])
            input_text = "[CLS]" + context + "[SEP]" + response + "[SEP]"
            input_tokenized = self.tokenizer.encode_plus(input_text, max_length=1024, padding='longest', return_tensors='pt')
            self.input_ids.append(input_tokenized['input_ids'])
            self.attention_masks.append(input_tokenized['attention_mask'])
            self.labels.append(torch.tensor(int(row["score"])))
        
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

data = pd.read_csv("data.csv", encoding='cp949')
tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
dataset = MyDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 토크나이저 및 모델 로드
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 모델 학습
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), inputs["labels"].view(-1))

        loss.backward()
        optimizer.step()

# 파인튜닝된 모델 저장
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_model")
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 데이터셋 로드
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        for idx, row in data.iterrows():
            context = str(row["context"])
            response = str(row["response"])
            input_text = "[CLS]" + context + "[SEP]" + response + "[SEP]"
            input_tokenized = self.tokenizer.encode_plus(input_text, max_length=1024, padding='longest', return_tensors='pt')
            self.input_ids.append(input_tokenized['input_ids'])
            self.attention_masks.append(input_tokenized['attention_mask'])
            self.labels.append(torch.tensor(int(row["score"])))
        
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

data = pd.read_csv("data.csv", encoding='cp949')
tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2", cache_dir="/path/to/local/directory")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
dataset = MyDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 토크나이저 및 모델 로드
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

# 모델 학습
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.shape[-1]), inputs["labels"].view(-1))

        loss.backward()
        optimizer.step()

# 파인튜닝된 모델 저장
model.save_pretrained("chatbot_model")
tokenizer.save_pretrained("chatbot_model")
