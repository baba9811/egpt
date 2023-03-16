import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling,Trainer, TrainingArguments, PreTrainedTokenizerFast

def load_dataset(file_name, encoding='cp949'):
    dataset = pd.read_csv(file_name, encoding=encoding)
    return dataset

def train(dataset):
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    dataset['input'] = dataset['context'] + ' ' + dataset['response']
    texts = dataset['input'].tolist()

    tokenized_texts = [tokenizer.encode(text) for text in texts]
    dataset = TextDataset(tokenizer, tokenized_texts, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == '__main__':
    dataset = load_dataset('data.csv', 'cp949')
    train(dataset)
