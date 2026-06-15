import torch
import pandas as pd
from transformers import AutoTokenizer
from model import EmotionConfig, calculate_max_seq_length, train, Transformer

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

vocabulary_size = tokenizer.vocab_size
hidden_size = 768
pad_token_id = tokenizer.pad_token_id
max_seq_length = calculate_max_seq_length(X_train, X_val, X_test, tokenizer)
num_of_classes = len(y_train.values.tolist())

config = EmotionConfig(src_vocab_size=vocabulary_size,
                       hidden_size=hidden_size,
                       num_attention_heads=4,
                       forward_intermediate_size=300,
                       hidden_dropout_prob=0.1,
                       max_seq_length=max_seq_length,
                       num_of_encode_layers=6,
                       num_of_classes=num_of_classes
                       )

transformer_model = Transformer(config)

train(transformer_model, X_train, y_train, tokenizer, pad_token_id, epochs=10, lr=0.001, bs=10, device=device)

print("prediction of i am so sad")
print(predict("I am so sad"))