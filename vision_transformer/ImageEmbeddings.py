import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
    """
    Embedding block of the transformer. 
    
    Takes an image as input and splits it on patches of size patch_size x patch_size. 
    Then for each patch it performs 

    Parameters
    ----------
    img_size : int
        size of image sides (both heigth and width)
    patch_size : int
        heigth and width of each patch that image would be splitted
    in_channels : int
        number of channels in input image
    d_model: int
        dimentionallity of the embeddings
    """
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.n_patches = (self.img_size // self.patch_size) ** 2

        self.embedder = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # (Batch, channels, heigth, width) --> (Batch, d_model, Seq^0.5, Seq^0.5)
        x = self.embedder(x)
        
        # (Batch, d_model, Seq^0.5, Seq^0.5) --> (Batch, d_model, Seq)
        x = x.flatten(2)        
        
        # (Batch, d_model, Seq) --> (Batch, Seq, d_model)
        x = x.transpose(-1, -2)
        return x
