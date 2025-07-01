import os
import pandas as pd
from datasets import Dataset
import evaluate
import torch
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np

# --- Config ---
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
DATA_DIR = "sentiment/data"
OUTPUT_DIR = "outputs/sentiment/saved_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3

# --- Load Data ---
def load_csv_as_dataset(split):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{split}.csv"))
    return Dataset.from_pandas(df)

train_dataset = load_csv_as_dataset("train")
val_dataset = load_csv_as_dataset("val")
test_dataset = load_csv_as_dataset("test")

# --- Tokenization ---
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def preprocess(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

cols = ['input_ids', 'attention_mask', 'label']
train_dataset.set_format(type='torch', columns=cols)
val_dataset.set_format(type='torch', columns=cols)
test_dataset.set_format(type='torch', columns=cols)

# --- Load Model ---
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# --- Metrics ---
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    print(f"Preds: {preds[:5]}")
    print(f"Labels: {labels[:5]}")
    acc = metric.compute(predictions=preds, references=labels)
    print(f"Accuracy metric result: {acc}")
    if acc is None or "accuracy" not in acc:
        return {"accuracy": 0.0}
    return {"accuracy": acc["accuracy"]}


# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    fp16=True if torch.cuda.is_available() else False,
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --- Train ---
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n--- Evaluating on test set ---")
test_results = trainer.evaluate(eval_dataset=test_dataset) # type: ignore
print(test_results)
