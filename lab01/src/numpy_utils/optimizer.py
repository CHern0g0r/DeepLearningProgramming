import numpy as np

from numpy_utils.layers import (
    get_paremeterized_layers
)


def zero_grad(layer):
    for param in layer.params:
        form = getattr(layer, param.grad)
        to = np.zeros_like(form)
        setattr(layer, param.grad, to)


class WeightBlock:
    def __init__(self, layer, param):
        self.layer = layer
        self.name = param.weight
        self.dname = param.grad

    def getv(self):
        return getattr(self.layer, self.name)

    def setv(self, v):
        setattr(self.layer, self.name, v)

    def getg(self):
        return getattr(self.layer, self.dname)

    def setg(self, v):
        setattr(self.layer, self.dname, v)


class NAdamOpt:
    def __init__(self, model):
        self.layers = get_paremeterized_layers(model)
        self.wblocks = [
            WeightBlock(la, pa)
            for la in self.layers
            for pa in la.params
        ]
        self.lr = 0.002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.md = 0.004
        self.t = 1
        self.musum = 0
        self.m0 = 0
        self.v0 = 0
        self.musum = 0

    def step(self):
        for wb in self.wblocks:
            w = wb.getv()
            g = wb.getg()
            d = self.get_d(w, g)
            new_w = w - d
            wb.setv(new_w)
        self.t += 1

    def get_d(self, w, g):
        mu_t = self.beta1 * (1 - np.power(0.96, self.t * self.md) / 2)
        mu_t1 = self.beta1 * (
            1 - np.power(0.96, (self.t + 1) * self.md) / 2
        )
        self.musum += mu_t
        mt = self.beta1 * self.m0 + (1 - self.beta1) * g
        vt = self.beta2 * self.v0 + (1 - self.beta2) * np.power(g, 2)
        m = (
            mu_t1 * mt / (1 - self.musum - mu_t1) +
            (1 - mu_t) * g / (1 - self.musum)
        )
        v = vt / (1 - np.power(self.beta2, self.t))
        dw = m / (np.sqrt(v) + self.eps)
        return self.lr * dw

    def zero_grad(self):
        for wb in self.wblocks:
            g = wb.getg()
            if g is not None:
                newg = np.zeros_like(g)
                wb.setg(newg)
