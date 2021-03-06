import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_L1', 'transform_train_optical_flow', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_L1', 'transform_test_optical_flow', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_flat_L1',
    '--optimizer', 'sgd_1_slow',
    '--scheduler', 'scheduler_3',
    '--num-epoch', "90",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
]

sys.argv = [sys.argv[0]] + args_list

import video_yyz.train
'''
 * Test Clip Acc@1 77.474
100%|█████████████████████████████████████████████████████████████████████| 90/90 [2:51:02<00:00, 114.03s/it]
Training time 2:51:02
'''