import copy
import torch

from torch import nn
from typing import List

from torch_utils.layers import (
    MBconfig,
    MBConv,
    ConvArgs,
    ConvBNSiLU
)


class EfficientNet(nn.Module):
    def __init__(self,
                 config_list: List[MBconfig],
                 out_fts: int = 1000,
                 stochastic_depth_prob: float = 0.2
                 ):
        super().__init__()

        self.sdp = stochastic_depth_prob
        self.backbone = self._create_from_configs(config_list)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(*[
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, out_fts, bias=True)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.clf(x)

        return x

    def _create_from_configs(self,
                             config_list: List[MBconfig],
                             last_channel: int = None) -> nn.Sequential:
        layers = []

        first_args = ConvArgs(
            3, config_list[0].input_channels, 3, 2
        )
        layers.append(
            ConvBNSiLU(first_args)
        )

        total_layers = sum(cfg.num_layers for cfg in config_list)
        cur_layer = 0
        for config in config_list:
            sublayer = []
            for i in range(config.num_layers):
                cfg = copy.copy(config)
                if sublayer:
                    cfg.input_channels = cfg.out_channels
                    cfg.stride = 1

                cfg.sd_prob = self.sdp * float(cur_layer) / total_layers
                block = MBConv(cfg)
                sublayer.append(block)
                cur_layer += 1
            layers.append(nn.Sequential(*sublayer))

        last_in = config_list[-1].out_channels
        last_out = (
            last_channel
            if last_channel is not None
            else 4 * last_in
        )

        last_args = ConvArgs(
            last_in,
            last_out,
            kernel_size=1,
        )

        layers.append(
            ConvBNSiLU(
                last_args
            )
        )

        return nn.Sequential(*layers)
