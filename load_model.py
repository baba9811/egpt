import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import os

# Load finetuned model
model = GPT2LMHeadModel.from_pretrained('./results')

# Define input text
input_text = input("enter the utterance:")

tokenizer = GPT2Tokenizer()

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate output sequence
output_ids = model.generate(
    input_ids=input_ids,
    max_length=20,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=1,
)

# Decode output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
