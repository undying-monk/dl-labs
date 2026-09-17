import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class ConvBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding: int | None = None):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

    
class ExtendBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ShrinkBlock(nn.Module):
    """Standard Convolution -> Batch Normalization -> Leaky ReLU block"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,64,3), # 256x256x64
            nn.MaxPool2d(2),    # 128x128x64

            ExtendBlock(64,128, 1), # 128x128x128
            nn.MaxPool2d(2),  # 64x64x128

            ExtendBlock(128,256,1), # 64x64x256
            nn.MaxPool2d(2),    # 32X32X256

            ExtendBlock(256,512,1), # 32X32X512
            nn.MaxPool2d(2),    # 16x16x512

            ExtendBlock(512,1024,1), # 16x16x1024
            nn.MaxPool2d(2),    # 8x8x1024

            ### Decoder
            ExtendBlock(512,1024,1), # 16x16x1024
            nn.MaxPool2d(2),    # 8x8x1024
        )

    def forward(self, x):
        return x


