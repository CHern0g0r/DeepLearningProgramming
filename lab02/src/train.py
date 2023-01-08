import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path

from model import (
    BaseModel,
    BaseModel1
)
from utils import train, DTS

def main(train_pth, test_pth, result_pth, cuda, bs=100):
    device = torch.device(f'cuda:{cuda}')

    trainset = DTS(train_pth)
    testset = DTS(test_pth)
    trian_loader = DataLoader(trainset, batch_size=bs)
    test_loader = DataLoader(testset, batch_size=bs)

    # model = BaseModel(nfts=1024)
    model = BaseModel1(nfts=1024)
    optimizer = torch.optim.AdamW(model.parameters())

    Path(result_pth).mkdir(parents=True, exist_ok=True)

    train(
        200,
        model,
        trian_loader,
        test_loader,
        optimizer,
        device,
        result_pth
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_pth')
    parser.add_argument('--test_pth')
    parser.add_argument('--result_pth')
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()
    main(**vars(args))