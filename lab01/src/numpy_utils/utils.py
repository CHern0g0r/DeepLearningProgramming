import numpy as np


def save_grad(lst):
    def hook(grad):
        lst.append(grad)
    return hook


def ttn(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().numpy()


def check_comb(torch_layers,
               numpy_layers,
               X, label=None,
               return_grad=False,
               verbose=True):

    assert (label is not None) >= return_grad

    show = lambda x: None
    if verbose:
        show = print

    Xts = [X]
    Xns = [X.detach().numpy()]

    labeln = None if label is None else label.numpy()

    gradst = []
    rest = X
    if return_grad:
        rest.register_hook(save_grad(gradst))
    for layer in torch_layers[:-1]:
        rest = layer(rest)
        if return_grad:
            rest.register_hook(save_grad(gradst))
        Xts += [rest]
    if label is None:
        rest = torch_layers[-1](rest)
    else:
        rest = torch_layers[-1](rest, label)
    Xts += [rest]

    resn = Xns[0]
    for layer in numpy_layers[:-1]:
        resn = layer(resn)
        Xns += [resn]
    if labeln is None:
        resn = numpy_layers[-1](resn)
    else:
        resn = numpy_layers[-1](resn, labeln)
    Xns += [resn]
    show('Forward is right', np.allclose(ttn(rest), resn))
    show('but', np.power(ttn(rest) - resn, 2).mean())

    result = (
        rest,
        resn,
        Xts,
        Xns
    )

    if return_grad:
        rest.backward()

        grad = numpy_layers[-1].backward()
        gradsn = [grad]
        for lyr in numpy_layers[-2::-1]:
            grad = lyr.backward(grad)
            gradsn.append(grad)

        result += (gradst, gradsn)

    return result
