import json
from datasets import Dataset
from transformers import AutoTokenizer
from config import MODEL_NAME

def load_domain_dataset():
    with open("data/domain_faq.json", "r") as f:
        data = json.load(f)

    data = data * 50

    inputs = []
    targets = []

    for item in data:
        inputs.append(f"Question: {item['question']}\nAnswer:")
        targets.append(item["answer"])

    dataset = Dataset.from_dict({
        "input_text": inputs,
        "target_text": targets
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch")

    return dataset, tokenizer
