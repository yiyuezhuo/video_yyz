from video_yyz.tools import SVD, resize_video, split_video, generate_index
from collections import namedtuple
from typing import List

def resize(basic_stem):

    Config = namedtuple('Config', ['root', 'target_root', 'split_root', 'scale'])

    basic = str(SVD / basic_stem)
    free = f'{basic}_free'
    free_split = f'{free}_split'

    config_list: List[Config] = [
        Config(
            root=basic,
            target_root=free,
            split_root=free_split,
            scale=(228, 128)
        )
    ]

    for config in config_list:
        print("Processing: ", config)
        resize_video(config.root, config.target_root, config.scale)
        split_video(config.target_root, config.split_root, 'image-%05d.jpg')
        generate_index(config.target_root)