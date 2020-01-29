from torchvision.datasets import Kinetics400
from torchvision.datasets.video_utils import VideoClips
import json
from pathlib import Path
import imageio
import os
import torch
import numpy as np


class VideoClipsFast(VideoClips):
    '''
    Read previous split frame image from disk without decoding video to speed up 
    data loading.

    template is template used by ffmpeg to split video into frames. Such as 
    image-%05d.jpg
    '''
    def __init__(self, *args, template, root, root_split, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.template = template
        self.root_split = root_split

    def get_clip(self, idx):
        if idx >= self.num_clips():
            raise IndexError("Index {} out of range "
                             "({} number of clips)".format(idx, self.num_clips()))
        video_idx, clip_idx = self.get_clip_location(idx)

        video_path = self.video_paths[video_idx]
        clip_pts = self.clips[video_idx][clip_idx]
        resampling_idx = self.resampling_idxs[video_idx][clip_idx]

        r_path = Path(video_path).relative_to(self.root)
        path_list = []
        im_list = []
        for idx in resampling_idx:
            name = self.template % (idx+1)
            p = Path(self.root_split) / r_path.with_suffix('') / name
            path_list.append(p)
            
            im = imageio.imread(p)
            im_list.append(im)
        
        video = torch.as_tensor(np.stack(im_list, 0))
        audio = None
        info = {}
        return video, audio, info, video_idx


class VideoDatasetFast(Kinetics400):
    '''
    Use index file to denote labels instead of file structure.

    index.json
        samples -> List[(video_path, label_int)]
        classes -> List[label_str]

    train.json, val.test are expected in training.

    path is path relative to root. For example, it's car/cat1.mp4 not /home/yyz/dataset/car/car1.mp4
    root will be added when creating instance.

    Other changes compared to Kinetics400:
        default extensions is replaced with mp4 from avi
        force specifying frame_rate (None mode in Kinetics400 is not valid)
    '''
    def __init__(self, root, *, index_path, root_split, template,
                 frames_per_clip, step_between_clips=1, frame_rate,
                 extensions=('mp4',), transform=None, _precomputed_metadata=None,
                 num_workers=0, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0):
        # super().__init__(root)

        self.index_path = index_path
        self.root_split = root_split
        self.template = template
        
        with open(index_path) as f:
            index = json.load(f)
            classes = index['classes']
            self.classes = classes
            self.samples = [(os.path.join(root, path), label) for path, label in index['samples']]

        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClipsFast(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,

            template=template,
            root=root,
            root_split=root_split
        )
        self.transform = transform


