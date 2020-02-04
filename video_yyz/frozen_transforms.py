from torchvision.transforms import Compose
import video_yyz.transforms as T

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])


def transform_train_reference():
    return Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        T.RandomHorizontalFlip(),
        normalize,
        T.RandomCrop((112, 112))
    ])


def transform_test_reference():
    return Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        normalize,
        T.CenterCrop((112, 112))
    ])


def transform_train_1():
    return Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 228)),
        T.RandomHorizontalFlip(),
        normalize,
        T.RandomCrop((112, 112))
    ])


def transform_test_1():
    return Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 228)),
        normalize,
        T.CenterCrop((112, 112))
    ])


def transform_test_1_left():
    return Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 228)),
        normalize,
        T.LeftCrop((112, 112))
    ])


def transform_test_1_right():
    return Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 228)),
        normalize,
        T.RightCrop((112, 112))
    ])


def transform_train_optical_flow():
    return Compose([
        T.transform_optical_flow_raw,
        T.Resize((128, 228)),
        T.RandomHorizontalFlip(),
        T.RandomCrop((112, 112))
    ])


def transform_test_optical_flow():
    return Compose([
        T.transform_optical_flow_raw,
        T.Resize((128, 228)),
        T.CenterCrop((112, 112))
    ])


def transform_identity():
    return Compose([])