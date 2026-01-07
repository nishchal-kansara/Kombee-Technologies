import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score

MODEL_PATH = "./results/checkpoint-6250"
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

dataset = load_dataset("imdb", split="test")

texts = dataset["text"][:200]
labels = dataset["label"][:200]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

preds = outputs.logits.argmax(dim=1).tolist()

acc = accuracy_score(labels, preds)
print("Evaluation accuracy", acc)
