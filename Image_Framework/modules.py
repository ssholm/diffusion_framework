import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels # embedding dimension
        self.size = size # Size of square image

        # Attention is all you need module
        # Embed using 4 heads
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)

        # Normalise dimensions of input size
        # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        self.ln = nn.LayerNorm([channels])

        # GELU: Guassian Error Linear Units
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Change format from (B,C,H,W) to (B,H * W,C)
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        # Normalise last layer - Channels
        x_ln = self.ln(x)
        # Multihead Attention
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # Add attention to input
        attention_value = attention_value + x
        # Some linear layers with residual connection
        attention_value = self.ff_self(attention_value) + attention_value
        # Tranform back to (B,C,H,W)
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        # If mid_channels is set then use that number of channels to go from in to out channels
        # Else set mid channels to out channels
        if not mid_channels:
            mid_channels = out_channels
        
        # Make a residual connection in the forward
        self.residual = residual

        # Groupnorm: Works like batchnorm, it just splits channels into n groups and normalises those instead
        # GELU: Guassian Error Linear Units
        # - https://pytorch.org/docs/stable/generated/torch.nn.GELU.html#gelu
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        # SILU: Sigmoid Linear Units -> x * sigmoid(x)
        # - https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        # Simple max pool with some convolutions
        x = self.maxpool_conv(x)
        # Embed the positional encoded timestep and repeat it for each image
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # Add embedded timestep to image
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html#upsample
        # Scale width and height dimensions by the factor
        # Use algorithm specified by the mode
        # align_corners preserves the corners value
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        # SILU: Sigmoid Linear Units -> x * sigmoid(x)
        # - https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        # Upsample the input
        x = self.up(x)
        # Concatenate the skip connection from downsampling with the input
        x = torch.cat([skip_x, x], dim=1)
        # Some simple convolutions
        x = self.conv(x)
        # Embed the positional encoded timestep and repeat it for each image
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # Add embedded timestep to image
        return x + emb
