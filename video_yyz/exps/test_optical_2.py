'''
5x lr compared to test_optical_1.py
'''
import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_L5', 'transform_train_optical_flow', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_L5', 'transform_test_optical_flow', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_flat_L5',
    '--optimizer', 'sgd_1_fast',
    '--scheduler', 'scheduler_1',
    '--num-epoch', "30",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
]

sys.argv = [sys.argv[0]] + args_list

import video_yyz.train
'''
Test: Total time: 0:01:10
 * Test Clip Acc@1 53.816
100%|█████████████████████████████████████████████████████████████████████| 30/30 [2:01:12<00:00, 242.41s/it]
Training time 2:01:12
'''