from . import utils
import time
import torch

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq,
                    *, logger: utils.LightLogger):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", group='train', logger=logger)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips per sec', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = 'Epoch: [{}]'.format(epoch)
    for video, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        video, target = video.to(device), target.to(device)
        output = model(video)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, = utils.accuracy(output, target, topk=(1, ))
        batch_size = video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['clips per sec'].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()


def evaluate(model, criterion, data_loader, device, *, logger: utils.LightLogger):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", group='test', logger=logger)
    header = 'Test:'
    with torch.no_grad():
        for video, target in metric_logger.log_every(data_loader, 100, header):
            video = video.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(video)
            loss = criterion(output, target)

            acc1, = utils.accuracy(output, target, topk=(1, ))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Clip Acc@1 {top1.global_avg:.3f}'
          .format(top1=metric_logger.acc1))
    return metric_logger.acc1.global_avg