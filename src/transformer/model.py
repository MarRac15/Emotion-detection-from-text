import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

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
        attn_output = self.attention(x,x,x) #here pass mask if necessary
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
        self.positional_encoding = PositionalEncoding(self.embed_size, config.max_seq_length)

        self.encode_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_of_encode_layers)])

        self.clasifier = EmotionClassifier(config)

    def forward(self, x):
        x = self.encoder_embedding(x) * sqrt(self.embed_size)
        x = self.positional_encoding(x)
        x = self.encode_layers(x)
        x = self.clasifier(x)
        return x


class Config():    #pass all variables for models
    def __init__(self, src_vocab_size, hidden_size, num_attention_heads, forward_intermediate_size, hidden_dropout_prob, max_seq_length, num_of_encode_layers, num_of_classes):
        super().__init__()     
        self.src_vocal_size = src_vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = forward_intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_seq_length = max_seq_length # find by the longest entry in database
        self.num_of_encode_layers = num_of_encode_layers
        self.num_of_classes = num_of_classes

