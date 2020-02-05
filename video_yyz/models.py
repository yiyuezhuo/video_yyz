'''
It is necessary to wrap model "constructor" to change `pretrained` logic. 
'''
from torchvision import models
from torchvision.models import video
from torchvision.models.utils import load_state_dict_from_url
# from torchvision.models.video.resnet import model_urls
from torch import nn
import torch


def r2plus1d_18(*, num_classes, pretrained=False, progress=True, **kwargs):
    '''
    Use pretrained model except fc layer, so that we can use different num_classes.
    '''
    model = video.r2plus1d_18(pretrained=False, progress=False, num_classes=num_classes, **kwargs)
    if pretrained:
        arch = 'r2plus1d_18'
        state_dict = load_state_dict_from_url(video.resnet.model_urls[arch],
                                              progress=progress)
        del state_dict['fc.weight']
        del state_dict['fc.bias']

        incompatible_key = model.load_state_dict(state_dict, strict=False)
        assert set(incompatible_key.missing_keys) == set(['fc.weight', 'fc.bias'])
    return model


class WordBag(nn.Module):
    '''
    Simple wrap for a frame oriented classifier.
    '''
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
    
    def forward(self, x):
        '''
        x: (batch_size, channel, depth, height, width)
        '''
        assert x.shape[2] == 1
        x = x.squeeze(2)
        return self.classifier(x)


class FlatModel(nn.Module):
    '''
    Flat depth dimension to channel dimension.
    Therefore we can use usual image classifier.
    '''
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
    
    def forward(self, x):
        batch_size, channel, depth, height, width = x.shape
        x = x.view(batch_size, channel*depth, height, width)
        return self.backend(x)


def resnet18_flat(*, num_classes, input_channel, pretrained=False, progress=True, **kwargs):
    if pretrained:
        raise ValueError("No natural pretrained model existed")
    backend = models.resnet18(pretrained=pretrained, progress=progress, num_classes=num_classes, **kwargs)
    backend.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = FlatModel(backend)
    return model


class ResNetFeatureExtractor(nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CNNWithLSTM(nn.Module):

    def __init__(self, *, feature_extractor, hidden_size, num_classes, cnn_output_size):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.lstm = nn.LSTM(cnn_output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        '''
        x: (batch_size, channel, depth, height, width)
        '''
        batch_size, channel, depth, height, width = x.shape
        frame_list = []
        for d in range(depth):
            frame = self.feature_extractor(x[:, :, d, :, :])  # (batch_size, out_feature)
            frame_list.append(frame)
        x = torch.stack(frame_list, 0)  # (seq_len, batch_size, out_feature)  # seq_len = depth
        seq, hidden_pair = self.lstm(x)  # seq: (seq_len, batch_size, hidden_size)
        out = self.linear(seq[-1])
        return out


def cnn_lstm(*, num_classes, pretrained=False, progress=True):
    cnn = models.resnet18(pretrained=pretrained, progress=progress)
    feature_extractor = ResNetFeatureExtractor(cnn)
    model = CNNWithLSTM(feature_extractor=feature_extractor, hidden_size=100, num_classes=num_classes, cnn_output_size=512)
    return model


def resnet18_word_bag(*, num_classes, pretrained=False, progress=True, **kwargs):
    model = models.resnet18(pretrained=False, progress=progress, num_classes=num_classes, **kwargs)
    if pretrained:
        arch = 'resnet18'
        state_dict = load_state_dict_from_url(models.resnet.model_urls[arch],
                                              progress=progress)
        del state_dict['fc.weight']
        del state_dict['fc.bias']

        incompatible_key = model.load_state_dict(state_dict, strict=False)
        assert set(incompatible_key.missing_keys) == set(['fc.weight', 'fc.bias'])
    model = FlatModel(model)
    return model
