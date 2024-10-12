import torch
import torch.nn as nn
from vision_transformer.LayerNorm import LayerNorm

class Encoder(nn.Module): 
    """
    Keeps stacked encoder blocks of the transformer.

    Parameters
    ----------
    layers : nn.ModuleList
        list of encoder blocks stacked on each other
    """
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        
        self.layers = layers
        self.layer_norm = LayerNorm()
        
    def forward(self, x): 
        for encoder_block in self.layers: 
            x = encoder_block(x)
            
        return self.layer_norm(x)
    