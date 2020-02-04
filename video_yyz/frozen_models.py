from . import models


def r2plus1d_18_1():
    return models.r2plus1d_18(num_classes=3, pretrained=True)


def resnet18_flat_L5():
    return models.resnet18_flat(num_classes=3, input_channel=5*2, pretrained=False)


def resnet18_flat_L1():
    return models.resnet18_flat(num_classes=3, input_channel=1*2, pretrained=False)


def cnn_lstm_1():
    return models.cnn_lstm(num_classes=3, pretrained=True)


def resnet18_word_bag():
    return models.resnet18_word_bag(num_classes=3, pretrained=True)