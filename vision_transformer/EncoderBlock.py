import torch
import torch.nn as nn

from vision_transformer.FeedForwardBlock import FeedForwardBlock
from vision_transformer.MultiHeadAttn import MultiHeadAttention
from vision_transformer.SkipConnectionBlock import SkipConnectionBlock


class EncoderBlock(nn.Module):
    """
    Encoder block of the transformer. 

    Parameters
    ----------
    attention_block : MultiHeadAttention
        multihead attention object
    feed_forward_block : FeedForwardBlock
        feed forward object
    dropout_rate : float
        probability to drop activation
    """

    def __init__(
        self,
        attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout_rate: float,
    ):
        super().__init__()
        
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.skip_attention_connections = SkipConnectionBlock(dropout_rate)
        self.skip_feed_forward_connections = SkipConnectionBlock(dropout_rate)

    def forward(self, x, mask=None):
        x = self.skip_attention_connections(x, lambda x: self.attention_block(x, x, x, mask))
        x = self.skip_feed_forward_connections(x, self.feed_forward_block)
        return x
