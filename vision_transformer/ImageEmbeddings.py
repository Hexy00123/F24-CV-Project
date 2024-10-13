# TODO: Implemet image patcher class that takes
# batch of images and cut it into sequence of patches
# (Batch, channels, height, width) -> (Batch, sequence_length, channels, height, width)


# TODO: implement a class that would take a sequence of
# image patches (Batch, sequence_length, channels, height, width) and return a sequence of
# image embeddings (Batch, sequence_length, embedding_size)

import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
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


if __name__ == '__main__': 
    img_batch = torch.rand((20, 3, 64, 64))
    print(img_batch.shape)
    
    embedder = PatchEmbedder(64, 16, 3, 512)
    print(embedder(img_batch).shape)