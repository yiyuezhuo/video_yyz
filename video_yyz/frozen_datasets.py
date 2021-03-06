from .dataset import Kinetics400Indexed
from pathlib import Path

def _video_dataset_1(root, transform, index, 
                     frames_per_clip=16,
                     step_between_clips=1,
                     frame_rate=25,
                     num_workers=0):
    return Kinetics400Indexed(
        root,
        Path(root) / index,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        frame_rate=frame_rate,
        transform=transform,
        num_workers=num_workers
    )

def full_video_dataset_1(root, transform):
    return _video_dataset_1(root, transform, "train_val.json")


def train_video_dataset_1(root, transform):
    return _video_dataset_1(root, transform, "train.json")


def test_video_dataset_1(root, transform):
    return _video_dataset_1(root, transform, "val.json")


def train_video_dataset_1_L5(root, transform):
    # Used to generated L=5 optical stack feature
    return _video_dataset_1(root, transform, "train.json", frames_per_clip=6)


def test_video_dataset_1_L5(root, transform):
    # Used to generated L=5 optical stack feature
    return _video_dataset_1(root, transform, "val.json", frames_per_clip=6)


def train_video_dataset_1_L1(root, transform):
    # Used to generated L=1 optical stack feature
    return _video_dataset_1(root, transform, "train.json", frames_per_clip=2)


def test_video_dataset_1_L1(root, transform):
    # Used to generated L=1 optical stack feature
    return _video_dataset_1(root, transform, "val.json", frames_per_clip=2)


def train_video_dataset_1_rgb(root, transform):
    # used by word bag model
    return _video_dataset_1(root, transform, "train.json", frames_per_clip=1)


def test_video_dataset_1_rgb(root, transform):
    # used by word bag model
    return _video_dataset_1(root, transform, "val.json", frames_per_clip=1)

# Use num_workers > 0 to speed up dataset building, may not work in windows
def train_video_dataset_1_fast(root, transform):
    return _video_dataset_1(root, transform, "train.json", num_workers=32)

def test_video_dataset_1_fast(root, transform):
    return _video_dataset_1(root, transform, "val.json", num_workers=32)

def train_video_dataset_1_rgb_fast(root, transform):
    # used by word bag model
    return _video_dataset_1(root, transform, "train.json", frames_per_clip=1, num_workers=32)

def test_video_dataset_1_rgb_fast(root, transform):
    # used by word bag model
    return _video_dataset_1(root, transform, "val.json", frames_per_clip=1, num_workers=32)
