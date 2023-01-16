import torch

from math import ceil
from torch import nn
from typing import Union, Optional
from dataclasses import dataclass, field, asdict
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    SiLU,
    AdaptiveAvgPool2d,
    Sigmoid
)


Sameseq = Union[int, tuple[int], list[int]]


@dataclass
class ConvArgs:
    in_channels: int = field()
    out_channels: int = field()
    kernel_size: Sameseq = field(1)
    stride: Sameseq = field(default=1)
    padding: Sameseq = field(default=0)
    dilation: Sameseq = field(default=1)
    bias: bool = field(default=True)


@dataclass
class MBconfig:
    expand_ratio: int = field()
    kernel: int = field()
    stride: int = field()
    input_channels: int = field()
    out_channels: int = field()
    num_layers: int = field()
    width_mult: float = field()
    depth_mult: float = field()

    @staticmethod
    def _make_divisible(self, v: float, divisor: int, min_value: Optional[int] = None) -> int:
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def adjust_channels(self, channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return self._make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(ceil(num_layers * depth_mult))

    def __post_init__(self):
        self.input_channels = self.adjust_channels(self.input_channels, self.width_mult)
        self.out_channels = self.adjust_channels(self.out_channels, self.width_mult)
        self.num_layers = self.adjust_depth(self.num_layers, self.depth_mult)


class StochDepth(nn.Module):
    def __init__(self, p: float = 0.2, mode: str = 'row') -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def _stochastic_depth(input: torch.Tensor, p: float, mode: str, training: bool = True) -> torch.Tensor:
        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        if mode not in ["batch", "row"]:
            raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
        if not training or p == 0.0:
            return input

        survival_rate = 1.0 - p
        if mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = torch.empty(size, dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return input * noise

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


class ConvBNSiLU(nn.Module):
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
                 inner_channels: int,
                 activation: nn.Module = SiLU,
                 scaler: nn.Module = Sigmoid):
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
        self.act = activation()
        self.scale = scaler()

    def forward(self, X):
        out = self.pool(X)
        out = self.cin(out)
        out = self.act(out)
        out = self.cout(out)
        out = self.scale(out)
        return out * X


class MBConv1(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class MBConv6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
