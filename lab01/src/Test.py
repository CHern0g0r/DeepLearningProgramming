import torch
import numpy as np
import faulthandler
from torch import nn
from torch_utils.models import EfficientNet
from torch_utils.layers import ConvBNSiLU
from numpy_utils.models import EfficientNetNumpy
from numpy_utils.layers import (
    Layer,
    CrossEntropyLoss,
    Conv2dNormActivationLayer,
    FlattenLayer,
    get_conv
)
from numpy_utils.utils import (
    check_comb,
    ttn
)
from common.configs import MBconfig, ConvArgs


def get_windows(input, output_size, kernel_size,
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


def test():
    W = np.random.rand(32, 3, 3, 3)
    dout = np.random.rand(8, 32, 128, 128)

    dout_windows = get_windows(
        dout, (8, 3, 256, 256), 3,
        padding=1, stride=1, dilate=1
    )
    rot_kern = np.rot90(W, 2, axes=(2, 3))

    db = np.sum(dout, axis=(0, 2, 3))
    dx = np.einsum('bohwkl,oikl->bihw', dout_windows, rot_kern)
    print(db.shape, dx.shape)

if __name__ == '__main__':
    e = EfficientNet([
        MBconfig(*args)
        for args in [
            (1, 3, 1, 32, 16, 1),
            (6, 3, 2, 16, 24, 2),
            (6, 5, 2, 24, 40, 2),
            (6, 3, 2, 40, 80, 3),
            (6, 5, 1, 80, 112, 3),
            (6, 5, 2, 112, 192, 4),
            (6, 3, 1, 192, 320, 1),
        ]
    ])
    en = EfficientNetNumpy([
        MBconfig(*args)
        for args in [
            (1, 3, 1, 32, 16, 1),
            (6, 3, 2, 16, 24, 2),
            (6, 5, 2, 24, 40, 2),
            (6, 3, 2, 40, 80, 3),
            (6, 5, 1, 80, 112, 3),
            (6, 5, 2, 112, 192, 4),
            (6, 3, 1, 192, 320, 1),
        ]
    ])
    print(type(e), isinstance(e, nn.Module))
    print(type(en), isinstance(en, Layer))

    crit = nn.CrossEntropyLoss(reduction='mean')
    critn = CrossEntropyLoss()

    X = torch.rand((8, 3, 256, 256), requires_grad=True)
    # X = torch.rand((8, 32, 128, 128), requires_grad=True)
    Xn = ttn(X)
    yt = torch.randint(0, 1000, (8,))
    ytn = ttn(yt)
    
    # print(y.shape)
    # print(yn.shape)

    first_args = ConvArgs(
        3, 32, 3, 2
    )
    cn = Conv2dNormActivationLayer(first_args)

    faulthandler.enable()
    # res = check_comb([e, crit], [cn, en, critn], X, yt, return_grad=True)




    X = torch.rand((4, 3, 256, 256), requires_grad=True)
    Xn = ttn(X)
    yt = torch.randint(0, 1000, (4,))
    ytn = ttn(yt)

    c = ConvBNSiLU(first_args)
    f = nn.Flatten()
    fn = FlattenLayer()
    print(c(X).shape)
    res = check_comb([c, f, crit], [cn, fn, critn], X, yt, return_grad=True)
