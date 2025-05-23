from transformers import AutoTokenizer, DistilBertForSequenceClassification,Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_fraud_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_fraud_model")

df = pd.read_csv(r'C:\Col_projects\SCT\Datasets\balanced_fraud_call.csv')
dataset = Dataset.from_pandas(df)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["conversation"], padding="max_length", truncation=True)

# Tokenize and split again (you should ideally save the original val split)
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.2)
val_data = dataset["test"]

training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_data,
)

metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

predictions = trainer.predict(val_data)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

probs = predictions.predictions[:, 1]  # Probabilities for the positive class
fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend()
plt.grid()
plt.show()

precision, recall, _ = precision_recall_curve(y_true, probs)

plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

print(classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"]))