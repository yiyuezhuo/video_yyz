from .scheduler import WarmupMultiStepLR
from torch.optim.lr_scheduler import StepLR

def scheduler_1(data_loader, optimizer):
    lr_warmup_epochs = 10
    lr_milestones = [20, 30, 40]
    lr_gamma = 0.1
    warmup_factor = 1e-5

    warmup_iters = lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=warmup_factor)
    return lr_scheduler


def scheduler_2(data_loader, optimizer):
    lr_warmup_epochs = 10
    lr_milestones = [20]
    lr_gamma = 0.1
    warmup_factor = 1e-5

    warmup_iters = lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=warmup_factor)
    return lr_scheduler


def scheduler_3(data_loader, optimizer):
    step_size_epoch = 1000 
    step_size = step_size_epoch * len(data_loader)  # a large value
    gamma = 0.1
    return StepLR(optimizer, step_size=step_size, gamma=gamma)