import torch
import torch.nn as nn
from vision_transformer.LayerNorm import LayerNorm


class SkipConnectionBlock(nn.Module):
    """
    Skip Connection block of the transformer.

    Allows to make skip connection by passing the layer that
    should be skipped in the forward method.

    Parameters
    ----------
    dropout_rate : float
        probability to drop activation
    """

    def __init__(self, dropout_rate):
        super().__init__()
        
        self.layer_norm = LayerNorm()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, inner_layer):
        return x + self.dropout(inner_layer(self.layer_norm(x)))
