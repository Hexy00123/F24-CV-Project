import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention block of the transformer

    Parameters
    ----------
    d_model : int
        dimentionality of the model (embedding size)
    h : int
        number of heads; note: it is mandatory to d_model be divisible by h 
    dropout_rate : float
        probability to drop activation
    """
    
    def __init__(self, d_model: int, h: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model 
        self.h = h
        assert self.d_model % self.h == 0, "it is mandatory to d_model be divisible by h"

        self.d_k = self.d_model // self.h 
        
        self.query_transform = nn.Linear(d_model, d_model)
        self.key_transform = nn.Linear(d_model, d_model)       
        self.value_transform = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)       
               
    def attention(self, query, key, value, mask): 
        # (Batch, h, Seq, Seq)
        attention_score = (query @ key.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None: 
            attention_score.masked_fill_(mask==0, -1e9)
        attention_score = attention_score.softmax(dim=-1)
        attention_score = self.dropout(attention_score)
        
        # (Batch, h, Seq, d_k). (Batch, h, Seq, Seq)
        return (attention_score @ value), attention_score
            
        
    def forward(self, q, k, v, mask=None):
        # (Batch, Seq, d_model) --> (Batch, Seq, d_model)
        query = self.query_transform(q)
        key = self.key_transform(k)
        value =  self.value_transform(v)
        
        # (Batch, Seq, d_model) --> (Batch, Seq, h, d_k) --> (Batch, h, Seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
                
        # (Batch, h, Seq, d_k). (Batch, h, Seq, Seq)
        x, attention_scores = self.attention(query=query, key=key, value=value, mask=mask)
        
        # (Batch, h, Seq, d_k) --> (Batch, Seq, h, d_k) --> (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
                
        # (Batch, Seq, d_model) --> (Batch, Seq, d_model)
        x = self.w_o(x)
        return x  