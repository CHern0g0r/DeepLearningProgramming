import torch
import numpy as np
import torchvision.transforms as T

from torch.utils.data import (
    DataLoader
)
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from data_utils.dataset import CarsDataset
from numpy_utils.utils import ttn
from tqdm import tqdm


def get_dataloader(img_path='../data/train',
                   ann_path='../data/train1.csv',
                   batch_size=32,
                   lendataset=8126
                   ):
    trans = [
        T.RandomHorizontalFlip(),
        T.RandomRotation(90),
        T.RandomApply([T.GaussianBlur(5)]),
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    testtrans = [
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    trans = T.Compose(trans)
    testtrans = T.Compose(testtrans)

    trainids, testids = train_test_split(
        list(range(lendataset)),
        test_size=.15
    )

    dataset = CarsDataset(
        img_path,
        ann_path,
        transforms=trans,
        idxs=trainids
    )
    testdataset = CarsDataset(
        img_path,
        ann_path,
        transforms=testtrans,
        idxs=testids
    )

    trainloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    testloader = DataLoader(
        testdataset,
        batch_size=batch_size,
        shuffle=True
    )

    return trainloader, testloader


def train_epoch(epoch,
                model,
                dataloader,
                criterion,
                optimizer,
                numpy=False):

    cum_loss = 0
    losses = []
    preds = np.array([])
    labels = np.array([])
    n = len(dataloader)
    get_pred = (
        np.argmax
        if numpy
        else lambda x, y: torch.max(x, y).indices
    )

    for i, data in tqdm(enumerate(dataloader),
                        desc=f'Training {epoch}',
                        total=n):
        imgs, target = data
        if numpy:
            imgs = ttn(imgs)
            target = ttn(target)

        optimizer.zero_grad()

        y = model(imgs)
        loss = criterion(y, target)
        pred = get_pred(y, 1)

        if numpy:
            grad0 = criterion.backward()
            model.backward(grad0)
            optimizer.step()
        else:
            loss.backward()
            optimizer.step()
            loss = loss.item()

        preds = np.concatenate([preds, pred])
        labels = np.concatenate([labels, target])
        cum_loss += loss * imgs.shape[0]
        losses += [loss]

    return losses, f1_score(labels, preds, average='macro')


def val_epoch(epoch,
              model,
              dataloader,
              criterion,
              numpy):

    cum_loss = 0
    losses = []
    preds = np.array([])
    labels = np.array([])
    n = len(dataloader)
    get_pred = (
        np.argmax
        if numpy
        else lambda x, y: torch.max(x, y).indices
    )

    for i, data in tqdm(enumerate(dataloader),
                        desc=f'Training {epoch}',
                        total=n):
        imgs, target = data
        if numpy:
            imgs = ttn(imgs)
            target = ttn(target)

        y = model(imgs)
        loss = criterion(y, target)
        pred = get_pred(y, 1)

        if not numpy:
            loss = loss.item()

        preds = np.concatenate([preds, pred])
        labels = np.concatenate([labels, target])
        cum_loss += loss * imgs.shape[0]
        losses += [loss]

    return losses, f1_score(labels, preds, average='macro')
