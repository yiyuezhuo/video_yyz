from .dataset import Kinetics400Indexed
from pathlib import Path

def _video_dataset_1(root, transform, index, 
                     frames_per_clip=16,
                     step_between_clips=1,
                     frame_rate=25):
    return Kinetics400Indexed(
        root,
        Path(root) / index,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        frame_rate=frame_rate,
        transform=transform,
        num_workers=0
    )

def full_video_dataset_1(root, transform):
    return _video_dataset_1(root, transform, "train_val.json")

def train_video_dataset_1(root, transform):
    return _video_dataset_1(root, transform, "train.json")

def test_video_dataset_1(root, transform):
    return _video_dataset_1(root, transform, "val.json")
