import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load data
df = pd.read_csv('data.csv', encoding='cp949')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', pad_token='<pad>')

# Preprocess data
inputs = []
labels = []
for index, row in df.iterrows():
    context = str(row['context'])  # Convert context to string
    response = str(row['response'])  # Convert response to string
    input_text = context + tokenizer.eos_token + response
    label_text = row['score']  # Convert score to string
    inputs.append(input_text)
    labels.append(label_text)

# Encode inputs and labels
input_ids = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors='pt')['input_ids']
labels = torch.tensor(labels, dtype=torch.float32)

# Create data collator
batch_size = 1
max_seq_length = 512
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=batch_size)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Create Trainer
training_args = TrainingArguments(
    output_dir='./results',               # output directory
    num_train_epochs=1,                   # total number of training epochs
    per_device_train_batch_size=batch_size,   # batch size per device during training
    per_device_eval_batch_size=batch_size,    # batch size for evaluation
    warmup_steps=500,                     # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                    # strength of weight decay
    logging_dir='./logs',                 # directory for storing logs
    logging_steps=1000,
    evaluation_strategy='steps',          # evaluation strategy to adopt during training
    save_strategy='steps',                # checkpoint saving strategy
    save_steps=1000,                      # frequency of saving checkpoints
    eval_steps=1000,                      # frequency of evaluation
    load_best_model_at_end=True,          # whether or not to load the best model found during training at the end
    metric_for_best_model='eval_loss',    # metric to use to evaluate the best model
    greater_is_better=False,              # whether the `metric_for_best_model` should be maximized or not
    report_to='tensorboard',
    gradient_accumulation_steps = 10000,
)
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=input_ids, 
    data_collator=data_collator
)

# Train model
trainer.train()
