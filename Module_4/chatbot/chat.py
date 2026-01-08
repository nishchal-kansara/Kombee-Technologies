import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_NAME, OUTPUT_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
model.eval()

print("Domain Chatbot Ready")
print("Type exit to quit")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        break

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=80,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Bot:", answer)
