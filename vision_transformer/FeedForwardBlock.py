import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    Feed forward block of the transformer

    Parameters
    ----------
    d_model : int
        dimentionality of the model (embedding size)
    d_ff : int
        dimentionality of the feed forward layer
    dropout_rate : float
        probability to drop activation
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            nn.ReLU(),
            
            nn.Dropout(dropout_rate), 
            
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.feed_forward(x)
