'''
YYZ video training

data path is specified by environ variable.
For train_video_dataset_1
'''
# Place logic here to raise Exception asap.
from os import environ

def _get_env(env_name):
    value = environ.get(env_name, None)
    if value is None:
        print(f"Set {env_name} env before running this script")
        exit()
    return value

train_data = _get_env("TRAIN_DATA")
test_data = _get_env("TEST_DATA")

print(f"TRAIN_DATA={train_data}")
print(f"TEST_DATA={test_data}")

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train", help="train data pipeline, expect (dataset, transform, collate, dataloader)", nargs=4, required=True)
parser.add_argument("--test", help="val data pipeline", nargs=4, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--optimizer", required=True)
parser.add_argument("--scheduler", required=True)
parser.add_argument("--num-epoch", type=int, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--resume", default='')
parser.add_argument("--start-epoch", default=0, type=int)
parser.add_argument('--output-dir', default='.', help='path where to save')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--tensorboard-comment', default='', help='suffix appended to tensorboard folder')
parser.add_argument('--debug', action='store_true', help="exit before training")

args = parser.parse_args()

import torch
import torchvision
from torch import nn

from . import frozen_datasets, frozen_transforms, frozen_collate, frozen_dataloader
from . import frozen_models, frozen_optimizer, frozen_scheduler

from . import utils
from .utils import LightLogger
from .train_utils import train_one_epoch, evaluate

import os
import time
import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

print(args)
print("torch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)
print("torchvision video backend: ", torchvision.get_video_backend())

device = torch.device(args.device)
num_epoch = args.num_epoch
print_freq = args.print_freq

torch.backends.cudnn.benchmark = True


def _read_pipeline(root, dataset_name, transform_name, collate_name, dataloader_name):
    dataset_builder = frozen_datasets.__dict__[dataset_name]
    transform_builder = frozen_transforms.__dict__[transform_name]
    collate_builder = frozen_collate.__dict__[collate_name]
    dataloader_builder = frozen_dataloader.__dict__[dataloader_name]

    transform = transform_builder()
    dataset = dataset_builder(root, transform)
    collate = collate_builder()
    dataloader = dataloader_builder(dataset, collate)

    return dataset, transform, collate, dataloader

print("Loading training data")

dataset_train, transform_train, collate_train, data_loader_train = _read_pipeline(train_data, *args.train)
dataset_test, transform_test, collate_test, data_loader_test = _read_pipeline(test_data, *args.test)

print("train dataset", dataset_train, len(dataset_train), len(data_loader_train))
print("test dataset", dataset_test, len(dataset_test), len(data_loader_test))

print("Creating model")
model_builder = frozen_models.__dict__[args.model]
model = model_builder()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_builder = frozen_optimizer.__dict__[args.optimizer]
optimizer = optimizer_builder(model.parameters())

scheduler_builder = frozen_scheduler.__dict__[args.scheduler]
scheduler = scheduler_builder(data_loader_train, optimizer)

start_epoch = args.start_epoch  # default: 0

if args.resume:
    print(f"Read resume {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    start_epoch = checkpoint['epoch'] + 1

print("Start TensorBoard")
writer = SummaryWriter(comment=args.tensorboard_comment)
logger = LightLogger(writer)

if args.debug:
    exit()

print("Start training")
start_time = time.time()
for epoch in tqdm(range(start_epoch, num_epoch)):
    train_one_epoch(model, criterion, optimizer, scheduler, data_loader_train,
                    device, epoch, print_freq, logger=logger)
    evaluate(model, criterion, data_loader_test, device=device, logger=logger)
    if args.output_dir:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'args': args}
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
