import numpy as np


class Layer:
    def __init__(self):
        self.name = self.__class__.__name__

    def forward(self, X):
        return X

    def backward(self, y):
        ...

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

        # https://stackoverflow.com/questions/72382568/vectorized-way-to-multiply-and-add-specific-axes-in-numpy-array-convolutional-l
        # x = np.transpose(X, [0, 2, 3, 1])
        # input_shape = x.shape
        # x_pad = self._pad(x, self.padding, 0)
        # input_pad_shape = x_pad.shape
        # # get the shapes
        # batch_size, h, w, Cin = input_shape
        # w1 = np.transpose(self.W, [2, 3, 1, 0])
        # fh, fw, _, _ = w1.shape
        # # calculate output sizes; only symmetric padding is possible
        # hout = (h + 2*self.padding - fh) // self.stride + 1
        # wout = (w + 2*self.padding - fw) // self.stride + 1
        # windows = self._windows(array=x_pad, stride_size=self.stride, filter_shapes=(fh, fw),
        #                 out_width=wout, out_height=hout) # 2D matrix with shape (batch_size, Hout, Wout, fh, fw, Cin)
        # out3 = np.tensordot(windows, w1, axes=([3,4,5], [0,1,2]))
        # # self.inputs = x_windows
        # out3 = np.transpose(out3, [0, 3, 2, 1])
        # out3 += self.b[None, :, None, None]

        

        # https://stackoverflow.com/questions/72840140/pytorch-conv2d-vs-numpy-results-are-different
        # padded = np.pad(
        #     X,
        #     pad_width=((0,), (0,), (self.padding,), (self.padding,)),
        #     mode='constant',
        #     constant_values=(0.,)
        # )
        # x = padded
        # w = self.W
        
        # b,  ci, hi, wi = x.shape
        # co, ci, hk, wk = w.shape
        # ho = np.floor(1 + (hi - hk) / self.stride).astype(int)
        # wo = np.floor(1 + (wi - wk) / self.stride).astype(int)
        # out2 = np.zeros((b, co, ho, wo), dtype=np.float32)
        
        # x = np.expand_dims(x, axis=1)
        # w = np.expand_dims(w, axis=0)
        # for bi in range(b):
        #     for i in range(ho):
        #         for j in range(wo):
        #             x_windows = x[
        #                 bi, :, :,
        #                 i * self.stride:i * self.stride + hk,
        #                 j * self.stride: j * self.stride + wk
        #             ]
        #             res = np.sum(x_windows * w, axis=(2, 3, 4))
        #             out2[bi, :, i, j] = res
        
        # out2 += self.b[None, :, None, None]

        return out  # , out2, out3

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

    def _windows(self, array, stride_size, filter_shapes, out_height, out_width):
        strides = (array.strides[0], array.strides[1] * stride_size, array.strides[2] * stride_size, array.strides[1], array.strides[2], array.strides[3])
        return np.lib.stride_tricks.as_strided(array, shape=(array.shape[0], out_height, out_width, filter_shapes[0], filter_shapes[1], array.shape[3]), strides=strides, writeable=False)

    def _pad(self, array, pad_size, pad_val):
        return np.pad(array, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=(pad_val, pad_val))


