'''
It is necessary to wrap model "constructor" to change `pretrained` logic. 
'''

from torchvision.models import video
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.video.resnet import model_urls


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
