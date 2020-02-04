import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_L5', 'transform_train_optical_flow', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_L5', 'transform_test_optical_flow', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_flat_L5',
    '--optimizer', 'sgd_1_slow',
    '--scheduler', 'scheduler_3',
    '--resume', 'test_optical_1.pth',
    '--num-epoch', "90",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
    '--reset-optimizer',
    '--reset-scheduler',
]

sys.argv = [sys.argv[0]] + args_list

import video_yyz.train

'''
'''

