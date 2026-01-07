from datasets import load_dataset
from transformers import AutoTokenizer

def load_imdb_dataset(model_name, max_length=128):
    dataset = load_dataset("imdb")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    tokenized_dataset = tokenized_dataset.rename_column(
        "label",
        "labels"
    )

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized_dataset
