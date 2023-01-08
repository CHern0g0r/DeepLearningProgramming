import torch

from torch import nn
from typing import Optional
from dataclasses import dataclass, field, asdict
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    SiLU,
    AdaptiveAvgPool2d,
    Sigmoid
)


Sameseq = Optional[int, tuple[int], list[int]]


@dataclass
class ConvArgs:
    in_channels: int = field()
    out_channels: int = field()
    kernel_size: Sameseq = field(1)
    stride: Sameseq = field(default=1)
    padding: Sameseq = field(default=0)
    dilation: Sameseq = field(default=1)
    bias: bool = field(default=True)


class Block(nn.Module):
    def __init__(self,
                 conv_args: ConvArgs,
                 activate: bool = True) -> None:
        super().__init__()
        self.conv = Conv2d(**asdict(conv_args))
        self.bn = BatchNorm2d(conv_args.out_channels)
        self.act = SiLU() if activate else None

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        if self.act is not None:
            X = self.act(X)
        return X


class SE(nn.Module):
    def __init__(self,
                 outer_channels: int,
                 inner_channels: int):
        super().__init__()
        cai = ConvArgs(
            outer_channels,
            inner_channels
        )
        cao = ConvArgs(
            inner_channels,
            outer_channels
        )
        self.pool = AdaptiveAvgPool2d(1)
        self.cin = Conv2d(**asdict(cai))
        self.cout = Conv2d(**asdict(cao))
        self.act = SiLU()
        self.scale = Sigmoid()

    def forward(self, X):
        pass


class MBConv1(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class MBConv6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
