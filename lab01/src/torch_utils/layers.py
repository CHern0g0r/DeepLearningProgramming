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
    kernel_size: Sameseq = field(default=1)
    stride: Sameseq = field(default=1)
    padding: Sameseq = field(default=None)
    dilation: Sameseq = field(default=1)
    bias: bool = field(default=False)
    groups: bool = field(default=1)

    def __post_init__(self):
        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2 * self.dilation


@dataclass
class MBconfig:
    expand_ratio: int = field()
    kernel: int = field()
    stride: int = field()
    input_channels: int = field()
    out_channels: int = field()
    num_layers: int = field()
    width_mult: float = field(default=1.0)
    depth_mult: float = field(default=1.0)
    sd_prob: float = field(default=0.2)

    @staticmethod
    def _make_divisible(v: float,
                        divisor: int,
                        min_value: Optional[int] = None
                        ) -> int:
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def adjust_channels(channels: int, width_mult: float,
                        min_value: Optional[int] = None) -> int:
        return MBconfig._make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(ceil(num_layers * depth_mult))

    def __post_init__(self):
        self.input_channels = self.adjust_channels(
            self.input_channels, self.width_mult
        )
        self.out_channels = self.adjust_channels(
            self.out_channels, self.width_mult
        )
        self.num_layers = self.adjust_depth(
            self.num_layers, self.depth_mult
        )


class StochDepth(nn.Module):
    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p

    def _stochastic_depth(self, X: torch.Tensor, p: float,
                          training: bool = True) -> torch.Tensor:
        if p < 0.0 or p > 1.0:
            raise ValueError("fuck")
        if not training or p == 0.0:
            return X

        survival_rate = 1.0 - p
        size = [X.shape[0]] + [1] * (X.ndim - 1)
        noise = torch.empty(size, dtype=X.dtype, device=X.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        return X * noise

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._stochastic_depth(X, self.p, self.training)


class ConvBNSiLU(nn.Module):
    def __init__(self,
                 conv_args: ConvArgs,
                 activate: bool = True,
                 bn: bool = True) -> None:
        super().__init__()
        self.conv = Conv2d(**asdict(conv_args))
        self.bn = (
            BatchNorm2d(conv_args.out_channels)
            if bn else nn.Identity()
        )
        self.act = SiLU() if activate else nn.Identity()

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.act(X)
        return X


class SE(nn.Module):
    def __init__(self,
                 outer_channels: int,
                 inner_channels: int,
                 activation: nn.Module = SiLU,
                 scaler: nn.Module = Sigmoid):
        super().__init__()

        # inner_channels = outer_channels // r
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


class MBConv(nn.Module):
    def __init__(self, config: MBconfig) -> None:
        super().__init__()

        self.use_res = (
            config.stride == 1 and
            config.input_channels == config.out_channels
        )

        layers = []
        expanded_channels = config.adjust_channels(
            config.input_channels,
            config.expand_ratio
        )
        if expanded_channels != config.input_channels:
            conv_args = ConvArgs(
                config.input_channels,
                expanded_channels
            )
            layers.append(
                ConvBNSiLU(conv_args)
            )

        conv_args = ConvArgs(
            expanded_channels,
            expanded_channels,
            kernel_size=config.kernel,
            stride=config.stride,
            groups=expanded_channels
        )
        layers.append(
            ConvBNSiLU(conv_args)
        )

        squeeze_channels = max(1, config.input_channels // 4)
        layers.append(SE(
            expanded_channels,
            squeeze_channels
        ))

        conv_args = ConvArgs(
            expanded_channels,
            config.out_channels,
            kernel_size=1
        )
        layers.append(ConvBNSiLU(conv_args))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochDepth(config.sd_prob)
        self.out_channels = config.out_channels

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        result = self.block(X)
        if self.use_res:
            result = self.stochastic_depth(result)
            result += X
        return result


class MBConv1(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class MBConv6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
