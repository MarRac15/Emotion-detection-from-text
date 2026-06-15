from sklearn.metrics import accuracy_score, f1_score
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from dataset import EmotionDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAIN_PATH = "resources/data/train.csv"
TEST_PATH = "resources/data/test.csv"
VAL_PATH = "resources/data/val.csv"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


#how does the tokenizer works:
print("tokenizer example: ")
text = train_df["content"].iloc[0]

encoding = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=32
)

print(tokenizer.convert_ids_to_tokens(encoding["input_ids"]))
print(encoding["attention_mask"])
print()


train_dataset = EmotionDataset(
    texts=train_df["clean_text"],
    labels=train_df["label_id"],
    tokenizer=tokenizer,
)

val_dataset = EmotionDataset(
    texts=val_df["clean_text"],
    labels=val_df["label_id"],
    tokenizer=tokenizer,
)


#dataloaders:
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


print("Rozmiar batchy: ")
batch = next(iter(train_loader))
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print()


#validation function:
def val_epoch(model, dataloader):

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #average for all batches:
    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(
        all_labels,
        all_preds
    )

    val_macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    return val_loss, val_acc, val_macro_f1




bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
bert_model.to(DEVICE)
epochs = 5
batch_size = 16

optimizer = AdamW(bert_model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)



#train loop:
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
best_f1 = 0
patience = 2
patience_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    bert_model.train()

    train_loss = 0

    for batch in tqdm(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        #for exploding gradients:
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        train_loss+=loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    #validation:
    val_loss, val_acc, val_f1 = val_epoch(bert_model, val_loader)
    print(f"Training loss: {avg_train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    print(f"Val macro F1: {val_f1:.4f}")


    #saving the best model:
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(
            bert_model.state_dict(),
            "src/bert/best_bert.pt"
        )
        patience_counter=0
    else:
        patience_counter+=1
        print(f"No improvement on epoch {epoch+1}")
        print(f"Patience: {patience_counter}/{patience}")

    #early stopping
    if patience_counter == patience:
        print("Early stopping!")
        break

