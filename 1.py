import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = pd.read_csv(data_path, encoding='cp949')
        self.tokenizer = tokenizer
        self.max_length = max_length

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
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    max_length = len(input_text) + 50
    output = model.generate(input_ids, max_length
