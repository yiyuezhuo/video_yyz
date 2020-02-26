'''
quicker version
'''

import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb', 'transform_train_1', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_rgb', 'transform_test_1', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_word_bag',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_4',
    '--num-epoch', "10",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
]

print("sys.argv before", sys.argv)
sys.argv = [sys.argv[0]] + args_list
print("sys.argv after", sys.argv)

import video_yyz.train

'''
0119

oops, forget to save the result. (Test console result output is truncated)

...
100%|█████████████████████████████████████████████████████████████████| 10/10 [6:58:42<00:00, 2512.24s/it]
Training time 6:58:42

(From TensorBoard)
Test: 81.48%, Train: 83.58%
'''