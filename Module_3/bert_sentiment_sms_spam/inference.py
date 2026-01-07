import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./bert_sms_model/checkpoint-837"  # or checkpoint path

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {0: "ham", 1: "spam"}

texts = [
    "Congratulations you won a free lottery ticket",
    "Are we meeting tomorrow evening",
    "Urgent call this number to claim prize"
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)

for text, prob in zip(texts, probs):
    spam_prob = prob[1].item()
    label = "spam" if spam_prob > 0.6 else "ham"

    print(text)
    print("Spam probability", round(spam_prob, 3))
    print("Prediction", label)
    print()