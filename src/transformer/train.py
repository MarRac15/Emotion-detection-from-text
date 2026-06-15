import torch
import pandas as pd
from transformers import AutoTokenizer
from model import EmotionConfig, calculate_max_seq_length

TRAIN_PATH = "resources/data/train.csv"
TEST_PATH = "resources/data/test.csv"
VAL_PATH = "resources/data/val.csv"

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

config = EmotionConfig(src_vocab_size=vocabulary_size,
                       hidden_size=hidden_size,
                       num_attention_heads=4,
                       forward_intermediate_size=300
                       hidden_dropout_prob=0.1,
                       max_seq_length=max_seq_length)

