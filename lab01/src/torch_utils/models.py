import torch

from torch import nn

from layers import (
    MBConv1,
    MBConv6
)


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
