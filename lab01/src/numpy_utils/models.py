import copy
import numpy as np

from typing import List

from common.configs import (
    ConvArgs,
    MBconfig
)
from numpy_utils.layers import (
    Layer,
    AdaptiveAvgPool2dLayer,
    SequentialLayer,
    DropoutLayer,
    LinearLayer,
    FlattenLayer,
    Conv2dNormActivationLayer,
    MBConvLayer,
    get_conv
)


class EfficientNetNumpy(Layer):
    def __init__(self,
                 config_list: List[MBconfig],
                 out_fts: int = 1000,
                 stochastic_depth_prob: float = 0.2
                 ):
        super().__init__()

        self.sdp = stochastic_depth_prob
        # first_args = ConvArgs(
        #     3, config_list[0].input_channels, 3, 2
        # )
        # self.in_layer = Conv2dNormActivationLayer(first_args)
        self.backbone = self._create_from_configs(config_list)
        self.pool = AdaptiveAvgPool2dLayer(1)
        self.flat = FlattenLayer()
        self.clf = SequentialLayer(*[
            DropoutLayer(p=0.2),
            LinearLayer(1280, out_fts)
        ])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x = self.in_layer(x)
        x = self.backbone(x)

        x = self.pool(x)
        x = self.flat(x)

        x = self.clf(x)

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.clf.backward(grad)
        grad = self.flat.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.backbone.backward(grad)
        # grad = self.in_layer.backward(grad)
        return grad

    def _create_from_configs(self,
                             config_list: List[MBconfig],
                             last_channel: int = None) -> SequentialLayer:
        layers = []

        # first_args = ConvArgs(
        #     3, config_list[0].input_channels, 3, 2
        # )
        # layers.append(
        #     Conv2dNormActivationLayer(first_args)
        # )

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
                block = MBConvLayer(cfg)
                sublayer.append(block)
                cur_layer += 1
            layers.append(SequentialLayer(*sublayer))

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
            Conv2dNormActivationLayer(
                last_args
            )
        )

        return SequentialLayer(*layers)
