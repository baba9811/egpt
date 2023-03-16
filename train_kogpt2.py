# train_kogpt2.py
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def main():
    data_file = "data.csv"
    encoding = "cp949"
    dataset = pd.read_csv(data_file, encoding=encoding)
    dataset["combined"] = dataset.apply(lambda row: f"{row.context} [SEP] {row.response}", axis=1)
    dataset["combined"].to_csv("train_data.txt", header=False, index=False)

    model_name = "skt/kogpt2-base-v2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SEP]"]})

    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train_data.txt",
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="kogpt2_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()
