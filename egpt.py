import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data.csv", encoding="cp949")

# Define tokenizer and tokenize dataset
tokenizer = GPT2TokenizerFast.from_pretrained("taeminlee/kogpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
train_encodings = tokenizer(df['context'].tolist(), truncation=True, padding=True)
train_labels = tokenizer(df['response'].tolist(), truncation=True, padding=True)

# Convert dataset to PyTorch tensors
class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = ChatbotDataset(train_encodings)
eval_dataset = ChatbotDataset(train_encodings)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=5,  # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=100,  # log & save weights each logging_steps
    load_best_model_at_end=True,  # load the best model when finished training
    metric_for_best_model='eval_loss',
    evaluation_strategy='steps',
    save_total_limit=5,  # limit the total amount of checkpoints to save
)

# Define trainer
model = GPT2LMHeadModel.from_pretrained("taeminlee/kogpt2")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'labels': torch.stack([f['input_ids'] for f in data])},
)

# Train model
trainer.train()

# Save model
model.save_pretrained('chatbot_model')

# Load model
model = GPT2LMHeadModel.from_pretrained('chatbot_model')
tokenizer = GPT2TokenizerFast.from_pretrained('taeminlee/kogpt2')

# Start chatbot conversation
while True:
    user_input = input("나: ")
    if user_input == 'exit':
        print("챗봇과의 대화를 종료합니다.")
        break
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    bot_output = model.generate(input_ids=input_ids, max_length=1000, pad_token_id=tokenizer.pad_token_id, do_sample=True, top_k=50, top_p=0.95)
    bot_response = tokenizer.decode(bot_output[0], skip_special_tokens=True)
    print("챗봇:", bot_response)
