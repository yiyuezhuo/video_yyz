'''
test_word_bag_7 with weakended random_crop
'''
import sys

from pathlib import Path
from datetime import datetime
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb_fast', 'transform_train_3', 'val2vl', 'video_random_3',
    '--test', 'test_video_dataset_1_rgb_fast', 'transform_test_3', 'val2vl', 'video_uniform_3',
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
Test:  [3300/3387]  eta: 0:00:17  loss: 0.0664 (0.6911)  acc1: 97.5000 (83.6905)  time: 0.2410  data: 0.0928  max mem: 1387
Test: Total time: 0:11:15
 * Test Clip Acc@1 83.747
100%|███████████████████████████████████████████████████████████████████████████████████| 15/15 [8:22:08<00:00, 2008.58s/it]
Training time 8:22:08
'''