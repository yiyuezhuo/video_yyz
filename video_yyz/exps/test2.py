import sys

args_list = [
    '--train', 'train_video_dataset_1', 'transform_train_1', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1', 'transform_test_1', 'val2vl', 'video_uniform_2',
    '--model', 'r2plus1d_18_1',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_1',
    '--num-epoch', "30",
]

print("sys.argv before", sys.argv)
sys.argv = [sys.argv[0]] + args_list
print("sys.argv after", sys.argv)

import video_yyz.train
