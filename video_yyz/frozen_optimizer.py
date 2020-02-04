import torch


def sgd_1(parameters):
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def sgd_1_fast(parameters):
    lr = 0.05
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def sgd_1_slow(parameters):
    lr = 0.001  # 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer
