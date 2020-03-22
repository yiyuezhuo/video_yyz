'''
test_word_bag_3 with weak random crop
'''
import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb_fast', 'transform_train_3', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_rgb_fast', 'transform_test_3', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_word_bag',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_4',
    '--num-epoch', "15",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
]

print("sys.argv before", sys.argv)
sys.argv = [sys.argv[0]] + args_list
print("sys.argv after", sys.argv)

import video_yyz.train
'''
'''