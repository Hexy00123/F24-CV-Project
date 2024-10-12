import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer norm block of the transformer.

    Parameters
    ----------
    eps : float
        constant for numerical stability (in order to make denominator be not too small)
    """

    def __init__(self, eps=10**-6) -> None:
        super().__init__()
        
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    