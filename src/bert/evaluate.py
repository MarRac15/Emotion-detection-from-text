import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

from src.bert.dataset import EmotionDataset

TRAIN_PATH = "resources/data/train.csv"
TEST_PATH = "resources/data/test.csv"
VAL_PATH = "resources/data/val.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=7
)

model.load_state_dict(torch.load("src/bert/best_bert.pt"))
model.to(DEVICE)
model.eval()

test_df = pd.read_csv(TEST_PATH)
test_dataset = EmotionDataset(
    texts=test_df["clean_text"],
    labels=test_df["label_id"],
    tokenizer=tokenizer,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


#evaluate on the test set
def evaluate_test(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


#for single predictions
def predict(text, model, tokenizer):
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    pred = torch.argmax(outputs.logits, dim=1).item()
    return pred



#
all_preds, all_labels = evaluate_test(model, test_loader)
print("Class report:")
print(classification_report(all_labels, all_preds))

print()
cm = confusion_matrix(all_labels, all_preds)
# print(confusion_matrix(y_test, y_pred))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()


#SIngle prediction:
text = "I love this scene when Kenobi shows up"
print(f"Example prediction: {text}")
print(predict(text, model, tokenizer))