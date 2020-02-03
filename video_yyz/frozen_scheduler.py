from .scheduler import WarmupMultiStepLR

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