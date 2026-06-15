import torch
import pandas as pd
from transformers import AutoTokenizer
from model import EmotionConfig, calculate_max_seq_length, train, Transformer
from model import EmotionDataset
from model import test_epoch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from model import collate_fn

TRAIN_PATH = "resources/data/train.csv"
TEST_PATH = "resources/data/test.csv"
VAL_PATH = "resources/data/val.csv"

def predict(text,tokenizer,model,device):
    sequence = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)

    output = model(sequence.to(device))
    prediction = output.argmax(axis=1).item()

    return prediction


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["clean_text"]
y_train = train_df["label_id"]

X_val = val_df["clean_text"]
y_val = val_df["label_id"]

X_test = test_df["clean_text"]
y_test = test_df["label_id"]

#tokenization

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

tokenizer.add_special_tokens({'pad_token':'[PAD]'})

vocabulary_size = len(tokenizer)
hidden_size = 680
pad_token_id = tokenizer.pad_token_id
max_seq_length = calculate_max_seq_length(X_train, X_val, X_test, tokenizer)
num_of_classes = len(set(y_train))




config = EmotionConfig(src_vocab_size=vocabulary_size,
                       hidden_size=hidden_size,
                       num_attention_heads=4,
                       forward_intermediate_size=300,
                       hidden_dropout_prob=0.1,
                       max_seq_length=max_seq_length,
                       num_of_encode_layers=6,
                       num_of_classes=num_of_classes
                       )

transformer_model = Transformer(config).to(device)


train(transformer_model, X_train, y_train, X_val,y_val,tokenizer, epochs=10, lr=0.0001, bs=64, device=device)

print("prediction of i am so sad")
print(predict("I am so sad",tokenizer,transformer_model,device))


test_dataset = EmotionDataset(X_test,y_test,tokenizer)
test_dataloader = DataLoader(test_dataset, num_workers=0, batch_size=64, collate_fn=collate_fn,shuffle=True)
criterion = torch.nn.CrossEntropyLoss()


all_preds, all_labels = test_epoch(transformer_model, test_dataloader, device, criterion)
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
print(predict(text, transformer_model, tokenizer,device))