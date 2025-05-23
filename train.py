import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv(r'C:\Col_projects\SCT\Datasets\balanced_fraud_call.csv')
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["conversation"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_function, batched=True)

dataset = dataset.train_test_split(test_size=0.2)
train_data, val_data = dataset["train"], dataset["test"]

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

# Save trained model
model.save_pretrained("fine_tuned_fraud_model")
tokenizer.save_pretrained("fine_tuned_fraud_model")
