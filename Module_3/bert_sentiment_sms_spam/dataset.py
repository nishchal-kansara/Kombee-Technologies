from datasets import load_dataset
from transformers import AutoTokenizer

def load_sms_dataset(model_name, max_length=128):
    raw_dataset = load_dataset("sms_spam")

    dataset = raw_dataset["train"].train_test_split(
        test_size=0.2,
        seed=42
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["sms"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["sms"]
    )

    tokenized = tokenized.rename_column("label", "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return tokenized