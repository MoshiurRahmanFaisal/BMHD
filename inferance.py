import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Enable tqdm for pandas
tqdm.pandas()

# =========================
# Model & Tokenizer
# =========================
MODEL_DIR = "./Model_Run/Bengali/best_ModernBERT/checkpoint-4272"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# =========================
# Labels
# =========================
label_classes = [
    "Addiction",
    "Alcoholism",
    "Anxiety",
    "Asperger's",
    "Bipolar",
    "Borderline Personality",
    "Depression",
    "Neutral",
    "Schizophrenia",
    "Self Harm",
    "Suicidal Thought"
]

# =========================
# Inference Function
# =========================
def predict_mental_state(text):
    if pd.isna(text) or str(text).strip() == "":
        return None

    inputs = tokenizer(
        str(text),
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class_id = torch.argmax(logits, dim=1).item()

    return label_classes[pred_class_id]

# =========================
# Load Excel
# =========================
input_excel_path = "Mental_Health_Master_File.xlsx"
output_excel_path = "prediction_output.xlsx"

df = pd.read_excel(input_excel_path)

# =========================
# Run Predictions with Progress Bar
# =========================
print("🔍 Predicting Bengali descriptions...")
df["Predicted_Bengali"] = df["Bengali_Description"].progress_apply(predict_mental_state)

print("🔍 Predicting English descriptions...")
df["Predicted_English"] = df["English_Description"].progress_apply(predict_mental_state)

# Rename actual column
df.rename(columns={"Mental_State": "Actual_Mental_State"}, inplace=True)

# =========================
# Save Output
# =========================
df.to_excel(output_excel_path, index=False)

print("✅ Inference complete!")
print(f"📁 Saved predictions to: {output_excel_path}")


# import os
# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.preprocessing import LabelEncoder

# # =========================
# # Paths & Data
# # =========================
# CHECKPOINT_DIR = "./Model_Run/Bengali/best_ModernBERT"
# TEST_FILE = "Dataset/mental_test_data.xlsx"
# TEXT_COL = "Description"
# LABEL_COL = "Mental_State"
# MODEL_NAME = "jhu-clsp/mmBERT-base"

# # =========================
# # Load Test Data
# # =========================
# test_df = pd.read_excel(TEST_FILE)
# label_encoder = LabelEncoder()
# test_df[LABEL_COL] = label_encoder.fit_transform(test_df[LABEL_COL])
# num_labels = len(label_encoder.classes_)

# # =========================
# # Tokenizer
# # =========================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# class BengaliDataset(torch.utils.data.Dataset):
#     def __init__(self, texts, labels):
#         self.encodings = tokenizer(
#             texts.tolist(),
#             truncation=True,
#             padding=True,
#             max_length=512
#         )
#         self.labels = labels.tolist()

#     def __getitem__(self, idx):
#         item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# test_dataset = BengaliDataset(test_df[TEXT_COL], test_df[LABEL_COL])

# # =========================
# # Metrics
# # =========================
# def compute_metrics(preds, labels):
#     pred_labels = np.argmax(preds, axis=1)
#     f1 = f1_score(labels, pred_labels, average="weighted")
#     acc = accuracy_score(labels, pred_labels)
#     return acc, f1

# # =========================
# # Scan Checkpoints
# # =========================
# best_f1 = 0.0
# best_checkpoint = None

# for folder in os.listdir(CHECKPOINT_DIR):
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, folder)
#     if os.path.isdir(checkpoint_path) and "checkpoint" in folder:
#         print(f"🔹 Evaluating {folder} ...")
#         model = AutoModelForSequenceClassification.from_pretrained(
#             checkpoint_path, num_labels=num_labels
#         )

#         trainer = Trainer(model=model, tokenizer=tokenizer)
#         predictions = trainer.predict(test_dataset)
#         preds = predictions.predictions
#         labels = test_df[LABEL_COL].values
#         _, f1 = compute_metrics(preds, labels)

#         print(f"   → Weighted F1: {f1:.4f}")
#         if f1 > best_f1:
#             best_f1 = f1
#             best_checkpoint = checkpoint_path

# print(f"\n✅ Best Checkpoint: {best_checkpoint}")
# print(f"🔥 Best Weighted F1: {best_f1:.4f}")
