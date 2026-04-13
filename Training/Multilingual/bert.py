# =========================
# 1. Imports
# =========================
import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# 2. GPU CHECK
# =========================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ GPU available:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("❌ GPU not available, using CPU")

# =========================
# 3. Load Data
# =========================
TRAIN_FILE = "Dataset/mental_train_data.xlsx"
TEST_FILE = "Dataset/mental_test_data.xlsx"

TEXT_COL = "Description"
LABEL_COL = "Mental_State"

train_df = pd.read_excel(TRAIN_FILE)
test_df = pd.read_excel(TEST_FILE)

# =========================
# 4. Label Encoding
# =========================
label_encoder = LabelEncoder()

train_df[LABEL_COL] = label_encoder.fit_transform(train_df[LABEL_COL])
test_df[LABEL_COL] = label_encoder.transform(test_df[LABEL_COL])

num_labels = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

# =========================
# 5. Load BanglaBERT
# =========================
MODEL_NAME = "google-bert/bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# 6. Dataset Class
# =========================
class BengaliDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=512
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BengaliDataset(train_df[TEXT_COL], train_df[LABEL_COL])
test_dataset = BengaliDataset(test_df[TEXT_COL], test_df[LABEL_COL])

# =========================
# 7. Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# =========================
# 8. Model Init (for HP tuning)
# =========================
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

# =========================
# 9. Hyperparameter Tuning
# =========================
training_args = TrainingArguments(
    output_dir="./Model_Run/English/bert_search",
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

best_run = trainer.hyperparameter_search(
    direction="maximize",
    n_trials=10,
    hp_space=lambda trial: {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1)
    }
)

print("🔥 Best Hyperparameters:", best_run.hyperparameters)

# =========================
# 10. Train Final Model
# =========================
best_args = TrainingArguments(
    output_dir="./Model_Run/English/best_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_run.hyperparameters["learning_rate"],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=best_run.hyperparameters["num_train_epochs"],
    weight_decay=best_run.hyperparameters["weight_decay"],
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    logging_dir="./logs"
)

final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

trainer = Trainer(
    model=final_model,
    args=best_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# =========================
# 11. Test Prediction
# =========================
predictions = trainer.predict(test_dataset)

y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_df[LABEL_COL].values

# =========================
# 12. Classification Report
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_
)

print("\n📊 Classification Report:\n")
print(report)

with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# =========================
# 13. Save Predictions to Excel
# =========================
test_df["Actual_Mental_State"] = label_encoder.inverse_transform(y_true)
test_df["Predicted_Mental_State"] = label_encoder.inverse_transform(y_pred)

OUTPUT_FILE = "./Model_Run/bert_test_predictions.xlsx"

test_df[
    ["Description", "Actual_Mental_State", "Predicted_Mental_State"]
].to_excel(OUTPUT_FILE, index=False)

print(f"\n✅ Predictions saved to {OUTPUT_FILE}")
