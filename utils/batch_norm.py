from torch import nn
import torch

"""
some functions adapted from
https://github.com/timgaripov/dnn-mode-connectivity/blob/master/utils.py
"""


def isbatchnorm(module):
    return issubclass(module.__class__, nn.modules.batchnorm._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def set_new_bn_momentum(model, new_momentum):
    for module in model.modules():
        if isbatchnorm(module):
            module.momentum = new_momentum


def update_bn(loader, model, device, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0

    for input, _ in loader:
        with torch.no_grad():
            input = input.to(device)
            batch_size = input.data.size(0)

            momentum = batch_size / (num_samples + batch_size)
            for module in momenta.keys():
                module.momentum = momentum

            model(input, **kwargs)
            num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
