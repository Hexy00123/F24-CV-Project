import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding block of the transformer.

    Adds positional encoding to a sequence of image embeddings.
    Ensures that each element in the sequence retains information about its relative position.

    Parameters
    ----------
    d_model : int
        dimentionality of the model (embedding size)

    max_len : int
        The maximum length of the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)

        # Calculate position values for each dimension of the positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register positional encodings as buffer (won't be updated during training)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input image embeddings.

        Args
        ----------
            x : torch.Tensor
                A tensor of shape (Batch, sequence_length, embedding_size),
                where Batch is the number of image sequences, sequence_length is the
                number of embeddings in each sequence, and embedding_size is the dimension of each embedding.

        Returns
        ----------
            torch.Tensor
                A tensor of the same shape as the input, but with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x
