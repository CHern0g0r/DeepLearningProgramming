import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, nfts=768, act=nn.Sigmoid) -> None:
        super().__init__()
        self.l1 = nn.Linear(nfts, 700)
        self.l2 = nn.Linear(700, 250)
        self.l3 = nn.Linear(250, 180)
        self.act = act()

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        return x


class BaseModel1(nn.Sequential):
    def __init__(self, nfts=768, act=nn.ReLU) -> None:
        layers = [
            nn.Linear(nfts, 1024),
            act(),
            nn.Linear(1024, 1024),
            act(),
            nn.Linear(1024, 512),
            act(),
            nn.Linear(512, 256),
            act(),
            nn.Linear(256, 180)
        ]
        super().__init__(*layers)
