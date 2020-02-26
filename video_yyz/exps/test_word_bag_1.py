import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb', 'transform_train_1', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_rgb', 'transform_test_1', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_word_bag',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_1',
    '--num-epoch', "30",
    '--tensorboard-comment', name,
    '--checkpoint-name', name + '.pth',
]

print("sys.argv before", sys.argv)
sys.argv = [sys.argv[0]] + args_list
print("sys.argv after", sys.argv)

import video_yyz.train
'''
 * Test Clip Acc@1 87.105
100%|████████████████████████████████████████████████████████████████████████| 30/30 [44:52<00:00, 89.74s/it]
Training time 0:44:52

0109 dataset

Test: Total time: 0:11:58
 * Test Clip Acc@1 85.294
100%|████████████████████████████████████████████████████████████████| 30/30 [21:01:40<00:00, 2523.36s/it]
Training time 21:01:40

0119
Test: Total time: 0:05:39
 * Test Clip Acc@1 91.188
100%|████████████████████████████████████████████████████████████████| 30/30 [10:03:11<00:00, 1206.40s/it]
Training time 10:03:11
'''