import torch

from torch import nn
from dataclasses import asdict
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    SiLU,
    AdaptiveAvgPool2d,
    Sigmoid
)

from common.configs import ConvArgs, MBconfig


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
