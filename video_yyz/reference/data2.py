'''
Here we assume video have been resized to value appearing in T.Resize.
'''

import torchvision
import video_yyz.transforms as T
from torchvision import get_video_backend

video_backend = get_video_backend()

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])

transform_train = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 228)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])

transform_test = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 228)),
    normalize,
    T.CenterCrop((112, 112))
])