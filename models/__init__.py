import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .preact_resnetwithswish import preact_resnetwithswish


from data import DATASETS


MODELS = ['resnet4', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'preact-resnet18-swish', 'preact-resnet34-swish']


def create_model(name, normalize, info):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
    Returns:
        torch.nn.Module.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if info['data'] in DATASETS:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        else:
            raise ValueError('Invalid model name {}!'.format(name))
    
    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))
        
    if normalize:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)
    
    model = torch.nn.DataParallel(model)
    model = model.to(device=device,memory_format=torch.channels_last)
    return model
