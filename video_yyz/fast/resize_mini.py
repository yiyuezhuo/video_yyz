'''
Generate small (960, 540), tiny (480, 270) version of original video (1920, 1080) to compare.
'''

from video_yyz.tools import SVD, resize_video, split_video, generate_index
from collections import namedtuple
from typing import List

Config = namedtuple('Config', ['root', 'target_root', 'split_root', 'scale'])

basic = str(SVD / 'video_sample_mini')
small = f'{basic}_small'
tiny = f'{basic}_tiny'
small_split = f'{small}_split'
tiny_split = f'{tiny}_split'

config_list: List[Config] = [
    Config(
        root=basic,
        target_root=small,
        split_root=small_split,
        scale=(960, 540)
    ),
    Config(
        root=basic,
        target_root=tiny,
        split_root=tiny_split,
        scale=(480, 270)
    )
]

for config in config_list:
    print("Processing: ", config)
    resize_video(config.root, config.target_root, config.scale)
    split_video(config.target_root, config.split_root, 'image-%05d.jpg')
    generate_index(config.target_root)