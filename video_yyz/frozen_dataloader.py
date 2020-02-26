from torchvision.datasets.samplers import UniformClipSampler, RandomClipSampler
import torch

def video_random_1(dataset, collate_fn):
    clips_per_video = 5
    batch_size = 5
    workers = 10
    sampler = RandomClipSampler(dataset.video_clips, clips_per_video)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn)
    return data_loader


def video_uniform_1(dataset, collate_fn):
    clips_per_video = 5
    batch_size = 5
    workers = 10
    sampler = UniformClipSampler(dataset.video_clips, clips_per_video)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn)
    return data_loader

def video_random_2(dataset, collate_fn):
    clips_per_video = 100
    batch_size = 5
    workers = 10
    sampler = RandomClipSampler(dataset.video_clips, clips_per_video)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn)
    return data_loader

def video_uniform_2(dataset, collate_fn):
    clips_per_video = 100
    batch_size = 5
    workers = 10
    sampler = UniformClipSampler(dataset.video_clips, clips_per_video)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn)
    return data_loader

# 4 gpu adaption + simplest model
def video_random_3(dataset, collate_fn):
    clips_per_video = 100
    batch_size = 80
    workers = 40
    sampler = RandomClipSampler(dataset.video_clips, clips_per_video)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn)
    return data_loader

def video_uniform_3(dataset, collate_fn):
    clips_per_video = 100
    batch_size = 80
    workers = 40
    sampler = UniformClipSampler(dataset.video_clips, clips_per_video)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn)
    return data_loader