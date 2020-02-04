import sys

args_list = [
    '--train', 'train_video_dataset_1_L5', 'transform_train_optical_flow', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_L5', 'transform_test_optical_flow', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_flat_L5',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_1',
    '--num-epoch', "30",
]

sys.argv = [sys.argv[0]] + args_list

import video_yyz.train

'''
 * Test Clip Acc@1 54.263
100%|█████████████████████████████████████████████████████████████████████| 30/30 [2:01:20<00:00, 242.69s/it]
Training time 2:01:20
'''