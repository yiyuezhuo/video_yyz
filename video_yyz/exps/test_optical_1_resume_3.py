import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_L5', 'transform_train_optical_flow', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_L5', 'transform_test_optical_flow', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_flat_L5',
    '--optimizer', 'sgd_1_slow',
    '--scheduler', 'scheduler_3',
    '--num-epoch', "90",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
    '--resume', 'test_optical_1.pth',
    '--reset-optimizer',
    '--reset-scheduler',
]

sys.argv = [sys.argv[0]] + args_list

import video_yyz.train

'''
Test: Total time: 0:01:11
 * Test Clip Acc@1 74.829
100%|█████████████████████████████████████████████████████████████████████| 60/60 [4:02:54<00:00, 242.91s/it]
Training time 4:02:54
'''

