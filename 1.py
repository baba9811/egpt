import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='cp949')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = str(self.data.loc[idx, 'context'])
        response = str(self.data.loc[idx, 'response'])
        inputs = self.tokenizer.encode_plus(context, response, add_special_tokens=True,
                                             max_length=self.max_length, truncation=True,
                                             padding='max_length', return_tensors='pt')
        score = self.data.loc[idx, 'score']
        score = np.array(score, dtype=np.float32)
        return inputs['input_ids'], inputs['attention_mask'], score


train_data_path = 'data.csv'
dataset = TextDataset(train_data_path, tokenizer, max_length=1024)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    max_length = len(input_text) + 50
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

for epoch in range(10):
    epoch_loss = 0
    model.train()
    for input_ids, attention_mask, score in dataloader:
        input_ids, attention_mask, score = input_ids.to(device), attention_mask.to(device), score.to(device)
        labels = input_ids.clone().detach()
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits.view(-1, tokenizer.vocab_size), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Epoch', epoch, 'loss:', epoch_loss / len(dataloader))

input_text = "안녕하세요?"
response = generate_response(input_text)
print(response)
