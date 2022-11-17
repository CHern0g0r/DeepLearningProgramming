import numpy as np

# from torch.nn import (
#     AdaptiveAvgPool2d,
#     BatchNorm2d,
#     SiLU,
#     Sigmoid,
#     Dropout,
#     CrossEntropyLoss
# )
# from torchvision.ops import StochasticDepth


def _sigmoid(X):
    return 1.0 / (1 + np.exp(-1 * X))


class Layer:
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = True

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

    def _init(self, *dims, mode=None):
        if mode == 'zeros':
            return np.zeros(dims)
        elif mode == 'uniform':
            return np.random.uniform(-1, 1, dims)
        elif mode == 'uniform01':
            return np.random.uniform(0, 1, dims)
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation - 1

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
        self.db, self.dw = None, None

    def forward(self, X):
        n, c, h, w = X.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        subm = self._get_windows(X, (n, c, out_h, out_w))

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
        rot_kern = np.rot90(self.W, 2, axes=(2, 3))

        self.db = np.sum(dout, axis=(0, 2, 3))
        self.dw = np.einsum('bihwkl,bohw->oikl', self.win, dout)
        dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)

        return dx

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


# Aux layers
class OneLayer(Layer):
    def forward(self, *X):
        return 1

    def backward(self, *grad):
        return 1


class FlattenLayer(Layer):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad


class SequentialLayer(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.sublayers = {
            f'l{i}': layer
            for i, layer in enumerate(layers)
        }

    def forward(self, X):
        for _, l in self.sublayers.items():
            X = l(X)
        return X

    def backward(self, grad):
        ...


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
