import math
import torch
import torch.nn as nn

from jaxtyping import Float
from torch import Tensor

class PositionalEncodings(nn.Module):
    """
    PositionalEncodings - adds positional encodings to the input tensor.

    Args:
        seq_len (int): The length of the sequence.
        d_model (int): The dimensionality of the model.

    """
    def __init__(self, 
                seq_len: int,
                d_model: int
    ):
        super().__init__()

        PE = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)
        
        PE = PE.unsqueeze(0)

        self.register_buffer('PE', PE)

    def forward(self, 
                x: Float[Tensor, "batch seq_len d_model"]
    ):
        x = x + self.PE[:, :x.size(1)]
        return x
