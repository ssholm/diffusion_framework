import torch
import torch.nn as nn
import torch_geometric.nn as pnn

from modules import *

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=1, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.ff = pnn.Sequential('x, t', [
            # Downsampling steps
            (DoubleConv(c_in, 64), 'x -> x1'),
            (Down(64, 128), 'x1, t -> x2'),
            (SelfAttention(128, 32), 'x2 -> x2'),
            (Down(128, 256), 'x2, t -> x3'),
            (SelfAttention(256, 16), 'x3 -> x3'),
            (Down(256, 256), 'x3, t -> x4'),
            (SelfAttention(256, 8), 'x4 -> x4'),
            
            # Middle steps
            (DoubleConv(256, 512), 'x4 -> x4'),
            (DoubleConv(512, 512), 'x4 -> x4'),
            (DoubleConv(512, 256), 'x4 -> x4'),

            # Upsampling steps
            (Up(512, 128), 'x4, x3, t -> x'),
            (SelfAttention(128, 16), 'x -> x'),
            (Up(256, 64), 'x, x2, t -> x'),
            (SelfAttention(64, 32), 'x -> x'),
            (Up(128, 64), 'x, x1, t -> x'),
            (SelfAttention(64, 64), 'x -> x'),
            (nn.Conv2d(64, c_out, kernel_size=1), 'x -> x')
        ])

        # Embedding of image labels
        self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        # https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
        # Positional encoding used for timestep t to be embedded in the image labels
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        # Encode timestep t
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Add labels to timestep encoding
        if y is not None:
            t += self.label_emb(y)

        # Feed images through network
        return self.ff(x, t)
