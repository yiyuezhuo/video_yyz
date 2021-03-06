import sys

from pathlib import Path
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1', 'transform_train_1', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1', 'transform_test_1', 'val2vl', 'video_uniform_2',
    '--model', 'cnn_lstm_1',
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
 * Test Clip Acc@1 84.329
100%|█████████████████████████████████████████████████████████████████████| 30/30 [3:37:42<00:00, 435.41s/it]
Training time 3:37:42
'''