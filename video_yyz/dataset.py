from torchvision.datasets import Kinetics400, VisionDataset
from torchvision.datasets.video_utils import VideoClips, unfold
import json
from pathlib import Path
import imageio
import os
import torch
import numpy as np
import math
import warnings


class VideoClipsFast(VideoClips):
    '''
    Read previous split frame image from disk without decoding video to speed up 
    data loading.

    template is template used by ffmpeg to split video into frames. Such as 
    image-%05d.jpg

    Override compute_clips_for_video and _resample_video_idx so that 
    resampling_idxs always be a tensor (advanced indexing) other than a slice.
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

    @staticmethod
    def compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate):
        if fps is None:
            # if for some reason the video doesn't have fps (because doesn't have a video stream)
            # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
            fps = 1
        if frame_rate is None:
            frame_rate = fps
        total_frames = len(video_pts) * (float(frame_rate) / fps)
        # idxs = VideoClips._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        idxs = VideoClipsFast._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        video_pts = video_pts[idxs]
        clips = unfold(video_pts, num_frames, step)
        if isinstance(idxs, slice):
            idxs = [idxs] * len(clips)
        else:
            idxs = unfold(idxs, num_frames, step)
        return clips, idxs

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        '''
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        '''
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs


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
        self.root = root  # overide Kinetics400 logic, but root is required by methods on VisionDataset

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


class Kinetics400Indexed(VisionDataset):
    """
    This class is similar to Kinetics400, but use a index file to build classes and samples,
    instead of building them from IO operation.
    """
    def __init__(self, root, index_path, *, frames_per_clip, step_between_clips, frame_rate,
                 extensions=('mp4',), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0):
        super(Kinetics400Indexed, self).__init__(root)

        self.index_path = index_path

        with open(index_path) as f:
            index = json.load(f)
            classes = index['classes']
            self.classes = classes
            self.samples = [(os.path.join(root, path), label) for path, label in index['samples']]

        '''
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        '''
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(
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
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            # ignore UserWarning: The pts_unit 'pts' gives wrong results and 
            # will be removed in a follow-up version. Please use pts_unit 'sec'.
            warnings.simplefilter("ignore")
            video, audio, info, video_idx = self.video_clips.get_clip(idx)
        target = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        # return video, audio, label
        return dict(video=video, audio=audio, target=target, video_idx=video_idx)
