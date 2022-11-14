import numpy as np

# from torch.nn import (
#     AdaptiveAvgPool2d,
#     BatchNorm2d,
#     SiLU,
#     Sigmoid,
#     Dropout
# )
# from torchvision.ops import StochasticDepth


class Layer:
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = True

    def forward(self, X):
        return X

    def backward(self, y):
        ...

    def train(self):
        self.train = True

    def eval(self):
        self.trian = False

    def _init(self, *dims, mode=None):
        if mode == 'zeros':
            return np.zeros(dims)
        return np.random.rand(*dims)

    def __call__(self, *args):
        return self.forward(*args)

    def __str__(self):
        return (
            self.name + '(' +
            ', '.join(
                f'{k}={v}'
                for k, v in self.__dict__.items()
                if k != 'name' and not isinstance(v, np.ndarray)
            ) +
            ')'
        )


class SequentialLayer(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.sublayers = {
            f'l{i}': layer
            for i, layer in enumerate(layers)
        }

    def forward(self, X):
        for ln, l in self.sublayers.items():
            X = l(X)
        return X

    def backward(self, y):
        ...


class LinearLayer(Layer):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self.W = self._init(self.output, self.input)
        self.b = self._init(self.output)

    def forward(self, X):
        res = np.dot(X, self.W.T) + self.b
        return res


class ConvLayer(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation-1

        self.b = self._init(
            out_channels,
            mode=None if bias else 'zeros'
        )
        self.W = self._init(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        )

    def forward(self, X):
        n, c, h, w = X.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        subm = self._get_windows(X, (n, c, out_h, out_w))

        out = np.einsum('bihwkl,oikl->bohw', subm, self.W)
        out += self.b[None, :, None, None]

        return out

    def _get_windows(self, input, output_size):
        if self.dilation != 0:
            input = np.insert(
                input,
                range(1, input.shape[2]),
                0, axis=2
            )
            input = np.insert(
                input,
                range(1, input.shape[3]),
                0, axis=3
            )

        if self.padding != 0:
            input = np.pad(
                input,
                pad_width=((0,), (0,), (self.padding,), (self.padding,)),
                mode='constant',
                constant_values=(0.,)
            )

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = input.shape
        batch_str, channel_str, kern_h_str, kern_w_str = input.strides

        return np.lib.stride_tricks.as_strided(
            input,
            (out_b, out_c, out_h, out_w,
             self.kernel_size, self.kernel_size),
            (batch_str, channel_str, self.stride * kern_h_str,
             self.stride * kern_w_str, kern_h_str, kern_w_str)
        )


# No overfitting Layers
class BatchNormLayer(Layer):
    def __init__(self):
        super().__init__()


class DropoutLayer(Layer):
    def __init__(self):
        super().__init__()


# Complex Layers
class Conv2dNormActivationLayer(Layer):
    def __init__(self):
        super().__init__()


class MBConvLayer(Layer):
    def __init__(self):
        super().__init__()


class SqueezeExcitationLayer(Layer):
    def __init__(self):
        super().__init__()


# WTF Layers
class StochasticDepthLayer(Layer):
    def __init__(self):
        super().__init__()


# Pooling Layers
class AdaptiveAvgPool2dLayer(Layer):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, X):
        ...


# Activation Layers
class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1.0 / (1 + np.exp(-1 * X))


class SiLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1.0 / (1 + np.exp(-1 * X)) * X
