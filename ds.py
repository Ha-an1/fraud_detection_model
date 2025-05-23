from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset (Assuming 'fraud_data.csv' has 'text' and 'label' columns)
dataset = load_dataset("csv", data_files=r'C:\Col_projects\SCT\Datasets\balanced_fraud_call.csv')

# Split into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.2)

train_data = dataset["train"]
val_data = dataset["test"]

def tokenize_function(examples):
    return AutoTokenizer(examples["conversation"], padding="max_length", truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)
