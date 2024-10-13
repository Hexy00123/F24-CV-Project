import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to a sequence of image embeddings.
    Ensures that each element in the sequence retains information about its relative position.
        
    Args:
        embedding_size (int): The dimension of each embedding vector (size of the feature vector for each position).
        max_len (int): The maximum length of the sequence.
    """
    def __init__(self, embedding_size: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        pe = torch.zeros(max_len, embedding_size)

        # Calculate position values for each dimension of the positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embedding_size)
        # Register positional encodings as buffer (won't be updated during training)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input image embeddings.
        
        Args:
            x (torch.Tensor): A tensor of shape (Batch, sequence_length, embedding_size),
                              where Batch is the number of image sequences, sequence_length is the
                              number of embeddings in each sequence, and embedding_size is the dimension of each embedding.
        
        Returns:
            torch.Tensor: A tensor of the same shape as the input, but with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x
