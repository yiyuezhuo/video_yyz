import torchvision
import video_yyz.transforms as T

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])

transform_train_reference = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])

transform_test_reference = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 171)),
    normalize,
    T.CenterCrop((112, 112))
])

transform_train_1 = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 228)),
    T.RandomHorizontalFlip(),
    normalize,
    T.RandomCrop((112, 112))
])

transform_test_1 = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((128, 228)),
    normalize,
    T.CenterCrop((112, 112))
])