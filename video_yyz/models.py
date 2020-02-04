'''
It is necessary to wrap model "constructor" to change `pretrained` logic. 
'''

from torchvision.models import video
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.video.resnet import model_urls
from torch import nn


def r2plus1d_18(*, num_classes, pretrained=False, progress=True, **kwargs):
    '''
    Use pretrained model except fc layer, so that we can use different num_classes.
    '''
    model = video.r2plus1d_18(pretrained=False, progress=False, num_classes=num_classes, **kwargs)
    if pretrained:
        arch = 'r2plus1d_18'
        state_dict = load_state_dict_from_url(model_urls[arch],
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
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
    
    def forward(self, x):
        batch_size, channel, depth, height, width = x.shape
        x = x.view(batch_size, channel*depth, height, width)
        return self.classifier(x)

