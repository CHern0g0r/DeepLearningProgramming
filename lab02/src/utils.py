import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score


def convert_to_np(item):
    return np.array(list(map(float, item[1:-1].split())))


def convert_to_torch(item):
    return torch.Tensor(list(map(float, item[1:-1].split())))


class DTS(Dataset):
    def __init__(self, pth) -> None:
        super().__init__()
        self.data = pd.read_csv(pth, converters={'emb': convert_to_torch})
        self.data = self.data[['id', 'emb', 'cls']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _, emb, cl = self.data.iloc[index]
        return emb, cl


def train_ep(n, model, dataloader, optimizer, device):
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for i, data in enumerate(tqdm(dataloader, desc=f'Epoch {n}')):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.cpu().item()

        if i % 500 == 0:
            print(
                f'Epoch {n}: loss = {running_loss / (i+1)}'
            )
    return model, (running_loss / len(dataloader))


def eval(model, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    running_loss = .0
    correct = 0
    total = 0
    preds = []
    lbls = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Validate'):
            inputs, labels = data
            lbls.append(labels.numpy())
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.cpu().item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            preds.append(predicted.cpu().numpy())
    preds = np.concatenate(preds)
    lbls = np.concatenate(lbls)
    skacc = accuracy_score(lbls, preds)
    acc = (correct / total)
    loss = (running_loss / len(dataloader))

    print(f'Acc: {acc}, SkAcc: {skacc}, Loss: {loss}')

    return acc, loss, skacc


def train(n, model, trainloader, testloader, optimizer, device, save_pth):
    model = model.to(device)
    best_acc = None
    accs = []
    trls = []
    tels = []
    npaccs = []
    for ep in range(n):
        model, tr_loss = train_ep(ep, model, trainloader, optimizer, device)

        acc, te_loss, npacc = eval(model, testloader, device)

        accs += [acc]
        trls += [tr_loss]
        tels += [te_loss]
        npaccs += [npacc]

        if best_acc is None or acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_pth, f'epoch{ep}.pth'))

    df = pd.DataFrame({
        'accuracy': accs,
        'sk_accuracy': npaccs,
        'train_loss': trls,
        'test_loss': tels
    })
    df.to_csv(os.path.join(save_pth, 'res.csv'), index=False)
