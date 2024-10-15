import torch
import torch.nn as nn
import torchinfo

from vision_transformer.blocks.ImageEmbeddings import PatchEmbedder
from vision_transformer.blocks.PositionalEncodings import PositionalEncoding
from vision_transformer.blocks.MultiHeadAttn import MultiHeadAttention
from vision_transformer.blocks.FeedForwardBlock import FeedForwardBlock
from vision_transformer.blocks.EncoderBlock import EncoderBlock
from vision_transformer.blocks.Encoder import Encoder


class ViT(nn.Module):
    """
    Vision transformer implementation.

    Parameters
    ----------
    img_size: int
        size of input image
    
    in_channels: int
        number of channels in input image
    
    patch_size: int = 8
        the size of patch into which the picture will be divided
    
    d_model: int = 768
        dimentionallity of the model - image embedding size
    
    dropout_rate: float = 0.35
        probability to drop activation
        
    n_encoder_blocks: int = 8
        number of encoding blocks that would be used
    
    n_heads: int = 12
        number of heads that MultiHeadAttention block will use
    
    ff_size: int = 2048
        size of feed forward layer of transformer
    """
    

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        patch_size: int = 8,
        d_model: int = 768,
        dropout_rate: float = 0.35,
        n_encoder_blocks: int = 8,
        n_heads: int = 12,
        ff_size: int = 2048,
    ):
        super().__init__()
        
        self.img_size: int = img_size,
        self.in_channels: int = in_channels,
        self.patch_size: int = patch_size,
        self.d_model: int = d_model,
        self.dropout_rate: float = dropout_rate,
        self.n_encoder_blocks: int = n_encoder_blocks,
        self.n_heads: int = n_heads,
        self.ff_size: int = ff_size,

        # Image embeddings initializaton
        self.embedder = PatchEmbedder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            d_model=self.d_model,
        )

        # Position encoder initialization
        self.position_encoder = PositionalEncoding(self.embedder.d_model, max_len=100)

        # Encoder initialization
        self.encoder = ViT.make_encoder(
            d_model=self.d_model,
            n_encoder_blocks=self.n_encoder_blocks,
            n_heads=self.n_heads,
            ff_size=self.ff_size,
            dropout_rate=self.dropout_rate,
        )

        # Classification tocken initialization
        self.cls_tocken = nn.Parameter(torch.zeros((1, 1, d_model)))

    @staticmethod
    def make_encoder(*_, d_model, n_encoder_blocks, n_heads, ff_size, dropout_rate):
        encoder_blocks = []

        for _ in range(n_encoder_blocks):
            attention_block = MultiHeadAttention(d_model, n_heads, dropout_rate)
            ff_block = FeedForwardBlock(d_model, ff_size, dropout_rate)

            encoder_block = EncoderBlock(attention_block, ff_block, dropout_rate)
            encoder_blocks.append(encoder_block)

        encoder_blocks = nn.ModuleList(encoder_blocks)
        return Encoder(encoder_blocks)

    def forward(self, x):
        embeddings = self.embedder(x)
        embeddings = self.position_encoder(embeddings)
        cls_tockens = self.cls_tocken.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tockens, embeddings), dim=1)

        encoded = self.encoder(x)
        image_embedding = encoded[:, 0, :]

        return image_embedding
