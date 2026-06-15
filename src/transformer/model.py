import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#dot product for attention
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k) 
                                                    #calculate dot product and normalize     
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)    #attention_wegiths * value_of_embedding

#attention head
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
    
    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs
    
#multihead dimension 
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

#feedforward - second block in transformer

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x
    
#inject position information 
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        max_seq_length = config.max_seq_length

        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

#encoder block

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_size = config.hidden_size
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):#add mask in necessary
        attn_output = self.attention(x) #here pass mask if necessary
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
  
class EmotionClassifier(nn.Module):
    def __init__(self, config, intermediate_size=50):
        super().__init__()
        embed_size = config.hidden_size
        num_of_classes = config.num_of_classes
        intermediate_size = embed_size // 2

        self.fc1 = nn.Linear(embed_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, num_of_classes)
    
    def forward(self, x):
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.hidden_size
        self.encoder_embedding = nn.Embedding(config.src_vocab_size, self.embed_size)
        self.positional_encoding = PositionalEncoding(config)

        self.encode_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_of_encode_layers)])

        self.clasifier = EmotionClassifier(config)

    def forward(self, x):
        x = self.encoder_embedding(x) * sqrt(self.embed_size)
        x = self.positional_encoding(x)
        for layer in self.encode_layers:
            x = layer(x)
        x = self.clasifier(x)
        return x

class EmotionConfig():    #pass all variables for models
    def __init__(self, src_vocab_size, hidden_size, num_attention_heads, forward_intermediate_size, hidden_dropout_prob, max_seq_length, num_of_encode_layers, num_of_classes):
        super().__init__()     
        self.src_vocab_size = src_vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = forward_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_seq_length = max_seq_length # find by the longest entry in database
        self.num_of_encode_layers = num_of_encode_layers
        self.num_of_classes = num_of_classes

class EmotionDataset(Dataset):
    def __init__(self, X_data, y_data, tokenizer):
        #tokenize and vectorize text
        self.text = X_data.values.tolist()
        self.tokenizer = tokenizer
        #load labels
        self.label = y_data.values.tolist()

    def __len__(self):
        return len(self.label)
    
    def get_sequence_token(self, idx):
        sequence = self.tokenizer.encode(self.text[idx])#check if correct
        len_seq = len(sequence)
        return sequence, len_seq
    
    def get_labels(self, idx):
      return self.label[idx]
    
    def __getitem__(self, idx):
      sequence, len_seq = self.get_sequence_token(idx)
      label = self.get_labels(idx)
      return sequence, label, len_seq 

from transformers import AutoTokenizer  
#check what pad_idx would be by adding it to tokenizer
tokenizer_temp = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

tokenizer_temp.add_special_tokens({'pad_token':'[PAD]'})
pad_token_id = tokenizer_temp.pad_token_id

def train(model, X_data, y_data, tokenizer, epochs, lr, bs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    train_dataset = EmotionDataset(X_data,y_data,tokenizer)
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=bs, collate_fn=collate_fn,shuffle=True)

    #training loop
    for epoch in range(epochs):
        total_loss_train = 0
        total_acc_train = 0   
        for train_sequence, train_label in tqdm(train_dataloader):

            # Model prediction
            predictions = model(train_sequence.to(device))
            labels = train_label.to(device)
            loss = criterion(predictions, labels)

            # Calculate accuracy and loss per batch
            correct = predictions.argmax(axis=1) == labels
            acc = correct.sum().item() / correct.size(0)
            total_acc_train += correct.sum().item()
            total_loss_train += loss.item()

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        print(f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f}')

def collate_fn(batch):

    sequences, labels, lengths = zip(*batch)
    max_len = max(lengths)

    for i in range(len(batch)):
        if len(sequences[i]) != max_len:
            for j in range(len(sequences[i]),max_len):
                sequences[i].append(pad_token_id)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def calculate_max_seq_length(X_train, X_val, X_test, tokenizer):
    text = X_train.values.tolist()
    text.extend(X_val.values.tolist())
    text.extend(X_test.values.tolist())
    text = tokenizer(text)
    text = text["input_ids"]
    max_length = 0
    for t in text:
        if len(t) > max_length:
            max_length  = len(t)
    return max_length


    