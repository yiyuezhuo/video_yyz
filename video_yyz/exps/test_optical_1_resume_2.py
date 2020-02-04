import sys

args_list = [
    '--train', 'train_video_dataset_1_L5', 'transform_train_optical_flow', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_L5', 'transform_test_optical_flow', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_flat_L5',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_2',
    '--resume', 'test_optical_1.pth',
    '--num-epoch', "60",
]

sys.argv = [sys.argv[0]] + args_list

import video_yyz.train

'''
 * Test Clip Acc@1 46.921
100%|█████████████████████████████████████████████████████████████████████| 30/30 [2:01:42<00:00, 243.42s/it]
Training time 2:01:42
'''