import torch
import numpy as np
import faulthandler
from torch import nn
from torch.optim import NAdam
from torch_utils.models import EfficientNet
from torch_utils.layers import ConvBNSiLU
from numpy_utils.models import EfficientNetNumpy
from numpy_utils.layers import (
    Layer,
    CrossEntropyLoss,
    Conv2dNormActivationLayer,
    FlattenLayer,
    get_conv,
    get_paremeterized_layers,
    BatchNorm2dLayer,
    traverse
)
from numpy_utils.utils import (
    check_comb,
    ttn
)
from numpy_utils.optimizer import NAdamOpt
from common.configs import MBconfig, ConvArgs
from train_utils.train import (
    train_epoch,
    val_epoch,
    get_dataloader
)


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

    layers = get_paremeterized_layers(en)
    for layer in layers:
        for param in layer.params:
            print(np.sum(getattr(layer, param.weight)))

    opt = NAdamOpt(en)

    y = en(Xn)
    loss = critn(y, ytn)
    grad = critn.backward()
    en.backward(grad)

    opt.step()

    for layer in layers:
        for param in layer.params:
            print(np.sum(getattr(layer, param.weight)))

    print(sum(
        1 for m in e.modules()
        if isinstance(m, (
            nn.Conv2d,
            nn.Linear,
            nn.BatchNorm2d
        ))
    ))

    # faulthandler.enable()
    # res = check_comb([e, crit], [en, critn], X, yt, return_grad=True)

    # X = torch.rand((4, 3, 256, 256), requires_grad=True)
    # Xn = ttn(X)
    # yt = torch.randint(0, 1000, (4,))
    # ytn = ttn(yt)

    # x1 = cn(Xn)
    # x2 = en(x1)
    # res = critn(x2, ytn)
    # grad0 = critn.backward()
    # grad1 = en.backward(grad0)
    # grad2 = cn.backward(grad1)
