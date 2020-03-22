'''
test_word_bag_3 with decreased memory overload
'''
import sys

from pathlib import Path
from datetime import datetime
name = Path(__file__).stem

args_list = [
    '--train', 'train_video_dataset_1_rgb_fast', 'transform_train_1', 'val2vl', 'video_random_2',
    '--test', 'test_video_dataset_1_rgb_fast', 'transform_test_1', 'val2vl', 'video_uniform_2',
    '--model', 'resnet18_word_bag',
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
new

Test: Total time: 0:06:49
 * Test Clip Acc@1 85.587
100%|███████████████████████████████████████████████████████████████████████████████████| 15/15 [5:54:38<00:00, 1418.59s/it]
Training time 5:54:38

new (reshuffled) 
Expect low score to prove it's possible to prevent overfitting

Test: Total time: 0:09:13
 * Test Clip Acc@1 48.656
100%|███████████████████████████████████████████████████████████████████████████████████| 15/15 [5:53:07<00:00, 1412.50s/it]
Training time 5:53:07
'''