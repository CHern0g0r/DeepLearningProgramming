import time
import pandas as pd

from argparse import ArgumentParser
from torch import nn
from torch.optim import NAdam

from torch_utils.models import EfficientNet
from numpy_utils.models import EfficientNetNumpy
from numpy_utils.layers import CrossEntropyLoss
from numpy_utils.optimizer import NAdamOpt
from train_utils.train import (
    get_dataloader,
    train_epoch,
    val_epoch
)
from common.configs import MBconfig


def savecsv(data, name, path):
    d = pd.DataFrame({
        name: data
    })
    d.to_csv(f'{path}/{name}.csv', index=False)


def main(numpy, epochs=100):
    if numpy:
        res_path = '../result/numpy'
        model = EfficientNetNumpy([
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
        ], out_fts=196)
        criterion = CrossEntropyLoss()
        optimizer = NAdamOpt(model)
    else:
        res_path = '../result/torch'
        model = EfficientNet([
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
        ], out_fts=196)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = NAdam(model.parameters())

    train_loader, test_loader = get_dataloader(
        img_path='../data/train',
        ann_path='../data/train1.csv'
    )

    results = {
        'tloss': [],
        'vloss': [],
        'tscore': [],
        'vscore': [],
        'time': []
    }

    for i in range(epochs):
        start = time.time()

        t_loss, t_f1 = train_epoch(
            i,
            model,
            train_loader,
            criterion,
            optimizer,
            numpy
        )

        v_loss, v_f1 = val_epoch(
            i,
            model,
            test_loader,
            criterion,
            numpy
        )

        end = time.time()

        results['vloss'] += v_loss
        results['tloss'] += t_loss
        results['vscore'] += [t_f1]
        results['tscore'] += [v_f1]
        results['time'] += [end - start]

        for n, val in results.items():
            savecsv(val, n, res_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--numpy', action='store_true')
    args = parser.parse_args()
    main(args.numpy)
