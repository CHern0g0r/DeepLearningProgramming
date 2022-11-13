import numpy as np


class Layer:
    def __init__(self):
        self.name = self.__class__.__name__

    def forward(self, X):
        return X

    def backward(self, y):
        ...

    def _init(self, *dims, mode=None):
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
        self.dilation = dilation

        self.b = (self._init(out_channels) if bias else None)
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

        padded = np.pad(
            X,
            pad_width=((0,), (0,), (self.padding,), (self.padding,)),
            mode='constant',
            constant_values=(0.,)
        )

        n_H = int((h - self.kernel_size + 2 * self.padding) / self.stride) + 1
        n_W = int((w - self.kernel_size + 2 * self.padding) / self.stride) + 1
        n_C = self.W.shape[0]

        Z = np.zeros((n, n_C, n_H, n_W))
        for i in range(n):
            image = padded[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vs = h * self.stride
                        ve = vs + self.kernel_size
                        hs = w * self.stride
                        he = hs + self.kernel_size
                        sls = image[:, vs:ve, hs:he]
                        s = np.multiply(sls, self.W[c, ...]) + self.b[c, ...]
                        Z[i, c, h, w] = np.sum(s)

        return out, Z

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
