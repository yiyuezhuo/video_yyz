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

'''
 * Clip Acc@1 84.539
100%|█████████████████████████████████████████████████████████████████████| 30/30 [7:07:47<00:00, 855.59s/it]
Training time 7:07:47
'''