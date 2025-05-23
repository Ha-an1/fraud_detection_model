import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

model_path = "fine_tuned_fraud_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

def predict_fraud(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Fraudulent Call 🚨" if prediction == 1 else "Safe Call ✅"

ques=input("Enter the conversation:")
sample_text = ques
result = predict_fraud(sample_text)
print(f"Prediction: {result}")
