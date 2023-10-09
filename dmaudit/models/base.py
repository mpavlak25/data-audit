import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import warnings



def get_model(arch, n_classes, pretrained=True):
    #filter out warning from using pretrained model instead of specified weights
    

    model_cls = getattr(models, arch)

    #get weights
    if 'resnet18' in arch:
        model = model_cls(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif 'convnext_tiny' in arch:
        model = model_cls(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    elif 'swin_t' in arch:
        model = model_cls(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = model_cls(pretrained=pretrained)

    # reshape the last layer
    if 'resnet18' in arch or 'resnet34' in arch:
        model.fc = nn.Linear(512, n_classes)
    elif 'resnet50' in arch or 'resnet101' in arch:
        model.fc = nn.Linear(2048, n_classes)
    elif 'vgg' in arch:
        model.classifier[6] = nn.Linear(4096, n_classes)
    elif 'densenet' in arch:
        model.classifier = nn.Linear(1024, n_classes)
    elif 'convnext_small' in arch or 'convnext_tiny' in arch:
        classifier_list = list(model.classifier.children())
        classifier_list[-1] = nn.Linear(in_features=768,out_features=n_classes,bias=True)
        model.classifier = nn.Sequential(*classifier_list)
    elif 'convnext_base' in arch:
        classifier_list = list(model.classifier.children())
        classifier_list[-1] = nn.Linear(in_features=1024,out_features=n_classes,bias=True)
        model.classifier = nn.Sequential(*classifier_list)
    elif 'swin_t' in arch:
        model.head = nn.Linear(in_features=768, out_features=n_classes, bias=True)
    elif 'vit_b' in arch:
        model.heads = nn.Sequential(OrderedDict([('head',nn.Linear(in_features=768,out_features=n_classes,bias=True))]))
    else:
        raise NotImplementedError(f'Arch {arch} currently not implemented')

    return model
