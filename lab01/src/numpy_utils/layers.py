import numpy as np

from itertools import product
from collections import namedtuple
from common.configs import MBconfig, ConvArgs

# from torch.nn import (
#     Linear,
#     AdaptiveAvgPool2d,
#     BatchNorm2d,
#     SiLU,
#     Sigmoid,
#     Dropout,
#     CrossEntropyLoss,
#     Module,
#     Conv2d,
#     Sequential
# )
# from torchvision.ops import StochasticDepth


def _sigmoid(X):
    return 1.0 / (1 + np.exp(-1 * X))


Param = namedtuple('parameter', ['weight', 'grad'])


class Layer:
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = True
        self.params = []
        self.sublayers = []

    def forward(self, X):
        return X

    def backward(self, grad):
        return grad * self.derivative()

    def train(self):
        self.train = True

    def eval(self):
        self.trian = False

    def derivative(self):
        return 1

    def zero_grad(self):
        for param, grad in self.params:
            setattr(self, grad, None)
        for sl in self.sublayers:
            getattr(self, sl).zero_grad()

    def _init(self, *dims, mode=None):
        if mode == 'zeros':
            return np.zeros(dims)
        elif mode == 'uniform':
            return np.random.uniform(-1, 1, dims)
        elif mode == 'uniform01':
            return np.random.uniform(0, 1, dims)
        elif mode == 'ones':
            return np.ones(dims)
        return np.random.rand(*dims)

    def _get_windows(self, input, output_size, kernel_size,
                     padding=0, stride=1, dilate=0):
        if dilate != 0:
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

        if padding != 0:
            input = np.pad(
                input,
                pad_width=((0,), (0,), (padding,), (padding,)),
                mode='constant',
                constant_values=(0.,)
            )

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = input.shape
        batch_str, channel_str, kern_h_str, kern_w_str = input.strides

        return np.lib.stride_tricks.as_strided(
            input,
            (out_b, out_c, out_h, out_w,
             kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str,
             stride * kern_w_str, kern_h_str, kern_w_str)
        )

    def __call__(self, *args):
        return self.forward(*args)

    def __str__(self):
        return (
            self.name + '(' +
            ', '.join(
                f'{k}={v}'
                for k, v in self.__dict__.items()
                if (
                    k not in ['name', 'params', 'sublayers'] and
                    not isinstance(v, np.ndarray)
                )
            ) +
            ')'
        )


class LinearLayer(Layer):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        self.W = self._init(self.output, self.input)
        self.b = self._init(self.output)
        self.inp = None
        self.dW = None
        self.db = None
        self.params = [
            Param('W', 'dW'),
            Param('b', 'db')
        ]

    def forward(self, X):
        res = np.dot(X, self.W.T) + self.b
        self.inp = X
        return res

    def backward(self, grad):
        self.dW = np.dot(grad.T, self.inp)
        self.db = np.sum(grad, axis=0)
        return np.dot(grad, self.W)


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
        self.fuck = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation - 1
        self.bias = bias

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
        self.win = None
        self.Xin = None
        self.db, self.dW = None, None
        self.params = [Param('W', 'dW')]
        if self.bias:
            self.params += [Param('b', 'db')]

    def forward(self, X):
        n, c, h, w = X.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        subm = self._get_windows(
            X, (n, c, out_h, out_w),
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation
        )

        out = np.einsum('bihwkl,oikl->bohw', subm, self.W)
        out += self.b[None, :, None, None]

        self.Xin = X
        self.win = subm

        return out

    def backward(self, dout):
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        dout_windows = self._get_windows(
            dout, self.Xin.shape, self.kernel_size,
            padding=padding, stride=1, dilate=self.stride - 1
        )
        if self.fuck:
            for pr in product(*(list(map(lambda x: reversed(list(range(x))), dout_windows.shape)))):
                print(pr)
                dout_windows[pr] += 0.002

        rot_kern = np.rot90(self.W, 2, axes=(2, 3))
        print('-'*30)
        print(self.W.shape, dout.shape, self.kernel_size, self.Xin.shape, padding, self.stride, sep='\n')

        self.db = np.sum(dout, axis=(0, 2, 3))
        self.dW = np.einsum('bihwkl,bohw->oikl', self.win, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        return dx


class DepthwiseConvLayer(ConvLayer):
    def __init__(self,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super().__init__(
            1,
            channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=False
        )
        self.c = channels

    def forward(self, X):
        n, c, h, w = X.shape
        assert c == self.c
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        subm = self._get_windows(
            X, (n, c, out_h, out_w),
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation
        )

        out = np.einsum('bohwkl,oikl->bohw', subm, self.W)

        self.Xin = X
        self.win = subm

        return out

    def backward(self, dout):
        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        dout_windows = self._get_windows(
            dout, self.Xin.shape, self.kernel_size,
            padding=padding, stride=1, dilate=self.stride - 1
        )
        rot_kern = np.rot90(self.W, 2, axes=(2, 3))

        self.dW = np.einsum('bohwkl,bohw->okl', self.win, dout)[:, None, :, :]
        dx = np.einsum('bohwkl,oikl->bohw', dout_windows, rot_kern)

        return dx


def get_conv(conv_args: ConvArgs):
    if conv_args.groups == conv_args.in_channels:
        return DepthwiseConvLayer(
            channels=conv_args.in_channels,
            kernel_size=conv_args.kernel_size,
            stride=conv_args.stride,
            padding=conv_args.padding,
            dilation=conv_args.dilation
        )
    else:
        return ConvLayer(
            in_channels=conv_args.in_channels,
            out_channels=conv_args.out_channels,
            kernel_size=conv_args.kernel_size,
            stride=conv_args.stride,
            padding=conv_args.padding,
            dilation=conv_args.dilation,
            bias=conv_args.bias
        )


# Aux layers
class OneLayer(Layer):
    def forward(self, *X):
        return 1

    def backward(self, *grad):
        return 1


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self.in_size = None

    def forward(self, X):
        self.in_size = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.in_size)


class SequentialLayer(Layer):
    def __init__(self, *layers, log=False):
        super().__init__()
        self.sublayers = layers
        self.log = log

    def forward(self, X):
        for layer in self.sublayers:
            X = layer(X)
        return X

    def backward(self, grad):
        for i, layer in enumerate(reversed(self.sublayers)):
            if self.log:
                print(i, layer.__class__.__name__)
            grad = layer.backward(grad)
        return grad

    def __str__(self):
        return '\n\t'.join([
            f'{self.name}(',
            ',\n\t'.join(
                str(layer)
                for layer in self.sublayers
            ),
            ')'
        ])


# No overfitting Layers
class BatchNorm2dLayer(Layer):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.running_mean = self._init(num_features, mode='zeros')
        self.running_var = self._init(num_features, mode='ones')
        self.num_batches_tracked = 0
        self.W = self._init(num_features)
        self.b = self._init(num_features)
        self.dW = self._init(num_features, mode='zeros')
        self.db = self._init(num_features, mode='zeros')
        self.inp = None
        self.params = [
            Param('W', 'dW'),
            Param('b', 'db')
        ]

    def forward(self, X):
        self._check_shape(X)
        self.inp = X

        exponential_average_factor = 0.0

        if self.train:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = (
                        1.0 / float(self.num_batches_tracked)
                    )
                else:
                    exponential_average_factor = self.momentum

        if self.train:
            mean = X.mean(axis=(0, 2, 3))
            var = X.var(axis=(0, 2, 3))
            n = np.prod(X.shape) / X.shape[1]
            self.running_mean = (
                exponential_average_factor * mean +
                (1 - exponential_average_factor) * self.running_mean
            )
            self.running_var = (
                exponential_average_factor * var * n / (n - 1) +
                (1 - exponential_average_factor) * self.running_var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        X = (
            (X - mean[None, :, None, None]) /
            (np.sqrt(var[None, :, None, None] + self.eps))
        )
        if self.affine:
            X = X * self.W[None, :, None, None] + self.b[None, :, None, None]

        return X

    def backward(self, grad):
        gamma = self.W[None, :, None, None]
        eps = self.eps
        B = self.inp.shape[0] * self.inp.shape[2] * self.inp.shape[3]

        mean = self.inp.mean(axis=(0, 2, 3), keepdims=True)
        variance = self.inp.var(axis=(0, 2, 3), keepdims=True)
        x_hat = (self.inp - mean) / (np.sqrt(variance + eps))

        dL_dxi_hat = grad * gamma
        dL_dvar = (-0.5 * dL_dxi_hat * (self.inp - mean)).sum(
            (0, 2, 3), keepdims=True
        ) * ((variance + eps) ** -1.5)
        dL_davg = (-1.0 / np.sqrt(variance + eps) * dL_dxi_hat).sum(
            (0, 2, 3), keepdims=True
        ) + (dL_dvar * (-2.0 * (self.inp - mean)).sum(
            (0, 2, 3), keepdims=True
        ) / B)

        dX = (
            (dL_dxi_hat / np.sqrt(variance + eps)) +
            (2.0 * dL_dvar * (self.inp - mean) / B) +
            (dL_davg / B)
        )
        dW = (grad * x_hat).sum((0, 2, 3), keepdims=True).squeeze()
        db = (grad).sum((0, 2, 3), keepdims=True).squeeze()
        self.dW += dW
        self.db += db
        return dX

    def _check_shape(self, X):
        assert len(X.shape) == 4
        assert X.shape[1] == self.num_features


class DropoutLayer(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, X):
        if self.train:
            self.mask = np.random.binomial(
                1, self.p, X.shape).astype(np.float32)
            self.mask *= 1. / (1. - self.p)
            return self.mask * X
        return X

    def backward(self, grad):
        return self.mask * grad / (1. - self.p)


# Complex Layers
class ResidualLayer(SequentialLayer):
    def __init__(self, *layers):
        super().__init__(*layers)
        self.X = None
        self.Fx = None

    def forward(self, X):
        self.X = X
        self.Fx = super().forward(X)
        return self.Fx + X

    def backward(self, grad):
        dFx = super().backward(grad)
        return grad + dFx


class Conv2dNormActivationLayer(Layer):
    def __init__(self,
                 conv_args: ConvArgs,
                 activate: bool = True,
                 bn: bool = True) -> None:
        super().__init__()
        self.conv = get_conv(conv_args)
        self.bn = (
            BatchNorm2dLayer(conv_args.out_channels)
            if bn else Layer()
        )
        self.act = SiLULayer() if activate else Layer()

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.act(X)
        return X

    def backward(self, grad):
        grad = self.act.backward(grad)
        grad = self.bn.backward(grad)
        grad = self.conv.backward(grad)
        return grad


class MBConvLayer(Layer):
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
                get_conv(conv_args)
            )

        conv_args = ConvArgs(
            expanded_channels,
            expanded_channels,
            kernel_size=config.kernel,
            stride=config.stride,
            groups=expanded_channels
        )
        layers.append(
            get_conv(conv_args)
        )

        squeeze_channels = max(1, config.input_channels // 4)
        layers.append(SqueezeExcitationLayer(
            expanded_channels,
            squeeze_channels
        ))

        conv_args = ConvArgs(
            expanded_channels,
            config.out_channels,
            kernel_size=1
        )
        layers.append(get_conv(conv_args))

        self.block = SequentialLayer(*layers)
        self.stochastic_depth = StochasticDepthLayer(config.sd_prob)
        self.out_channels = config.out_channels

    def forward(self, X: np.ndarray) -> np.ndarray:
        result = self.block(X)
        if self.use_res:
            result = self.stochastic_depth(result)
            result += X
        return result

    def backward(self, grad):
        if not self.use_res:
            return self.block.backward(grad)
        # dFx = super().backward(grad)
        # return grad + dFx
        newgrad = self.stochastic_depth.backward(grad)
        dFx = self.block.backward(newgrad)
        return grad + dFx
        


class SqueezeExcitationLayer(SequentialLayer):
    def __init__(self, output, hidden,
                 act=None, scale=None):
        super().__init__(
            AdaptiveAvgPool2dLayer(1),
            ConvLayer(output, hidden, 1),
            (SiLULayer() if act is None else act()),
            ConvLayer(hidden, output, 1),
            (SigmoidLayer() if scale is None else scale())
        )
        self.y = None
        self.X = None

    def forward(self, X):
        self.X = X
        self.y = super().forward(X)
        return X * self.y

    def backward(self, grad):
        # x f'(x) + f(x)
        # grad * x * f'(x) + grad * f(x)
        # grad * f'(x) = super().backward(grad)
        grady = super().backward(grad)
        gradyx = self.X * grady
        grady2 = grad * self.y
        return gradyx + grady2



# WTF Layers
class StochasticDepthLayer(Layer):
    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.p = p
        self.noise = None

    def _stochastic_depth(self, X: np.ndarray, p: float,
                          training: bool = True) -> np.ndarray:
        if p < 0.0 or p > 1.0:
            raise ValueError("fuck")
        if not training or p == 0.0:
            return X

        survival_rate = 1.0 - p
        size = [X.shape[0]] + [1] * (X.ndim - 1)
        noise = np.random.binomial(1, survival_rate, size).astype(np.float64)
        if survival_rate > 0.0:
            noise /= survival_rate
        self.noise = noise
        return X * noise

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self._stochastic_depth(X, self.p, self.train)

    def backward(self, grad):
        return grad * self.noise


# Pooling Layers
class AvgPool2dLayer(Layer):
    def __init__(self,
                 kernel_size=1,
                 stride=None,
                 padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = (
            stride
            if stride is not None
            else kernel_size
        )
        self.padding = padding
        self.inp_shape = None

    def forward(self, X):
        b, c, h, w = self.inp_shape = X.shape

        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        wins = self._get_windows(
            X,
            (b, c, out_h, out_w),
            self.kernel_size,
            self.padding,
            self.stride,
            dilate=0
        )
        means = np.mean(wins, axis=(4, 5))
        return means

    def backward(self, grad):
        grad *= 1. / self.kernel_size**2
        b, c, h, w = grad.shape
        pre_grad = np.zeros(self.inp_shape)
        wins = self._get_windows(
            pre_grad,
            grad.shape,
            self.kernel_size
        )
        wins += grad[:, :, :, :, None, None]
        return pre_grad


class AdaptiveAvgPool2dLayer(AvgPool2dLayer):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.padding = 0

    def forward(self, X):
        b, c, h, w = X.shape
        self.stride = h // self.output_size
        self.kernel_size = h - (self.output_size - 1) * self.stride
        return super().forward(X)

    def backward(self, grad):
        return super().backward(grad)


# Activation Layers
class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X):
        self.X = X
        return _sigmoid(X)

    def derivative(self):
        sig = _sigmoid(self.X)
        return sig * (1 - sig)


class SiLULayer(Layer):
    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X):
        self.X = X
        return 1.0 / (1 + np.exp(-X)) * X

    def derivative(self):
        sig = _sigmoid(self.X)
        deriv = sig * (1 + self.X * (1 - sig))
        return deriv


class SoftmaxLayer(Layer):
    def forward(self, x):
        self.old_y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        return self.old_y

    def backward(self, grad):
        return self.old_y * (
            grad - (grad * self.old_y).sum(axis=1)[:, None]
        ) / self.old_y.shape[0]


# Criterion
class CrossEntropyCost(Layer):
    def __init__(self, reduction_fn=np.mean):
        super().__init__()
        self.reduction_fn = reduction_fn

    def forward(self, x, y):
        self.old_x = x.clip(min=1e-8, max=None)
        targets = np.zeros_like(x)
        targets[np.arange(len(x)), y] = 1
        self.old_y = targets
        whered = np.where(targets == 1, -np.log(self.old_x), 0)
        sumed = whered.sum(axis=1)
        return self.reduction_fn(sumed)

    def backward(self):
        return np.where(self.old_y == 1, -1 / self.old_x, 0)


class CrossEntropyLoss(Layer):
    def __init__(self, reduction_fn=np.mean):
        super().__init__()
        self.reduction_fn = reduction_fn
        self.sm = None
        self.targets = None
        self.ls = None

    def forward(self, X, y):
        targets = np.zeros_like(X)
        targets[np.arange(len(X)), y] = 1

        sm = np.exp(X) / np.sum(np.exp(X), axis=1)[:, None]
        out = np.sum(-targets * np.log(sm), axis=1)

        self.sm = sm
        self.targets = targets
        return self.reduction_fn(out)

    def backward(self):
        return (self.sm - self.targets) / self.sm.shape[0]
