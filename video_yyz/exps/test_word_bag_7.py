'''
test_word_bag_3 with increased num_classes
'''
import sys

from pathlib import Path
from datetime import datetime
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb_fast', 'transform_train_1', 'val2vl', 'video_random_3',
    '--test', 'test_video_dataset_1_rgb_fast', 'transform_test_1', 'val2vl', 'video_uniform_3',
    '--model', 'resnet18_word_bag_2',
    '--optimizer', 'sgd_1',
    '--scheduler', 'scheduler_4',
    '--num-epoch', "15",
    '--tensorboard-comment', name,
    '--checkpoint-name', f'{name}_{datetime.now()}.pth',
]

print("sys.argv before", sys.argv)
sys.argv = [sys.argv[0]] + args_list
print("sys.argv after", sys.argv)

import video_yyz.train

'''
(For furnace 2 data)
Test: Total time: 0:10:51
 * Test Clip Acc@1 80.673
100%|███████████████████████████████████████████████████████████████████████████████████| 15/15 [9:49:58<00:00, 2359.91s/it]
Training time 9:49:58
'''