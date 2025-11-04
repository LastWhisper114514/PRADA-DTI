import os
import importlib

import math
import torch
import torch.nn as nn

def get_all_backbones():
    return [backbone.split('.')[0] for backbone in os.listdir('backbones')
            if not backbone.find('__') > -1 and 'py' in backbone]

names = {}
backbones = get_all_backbones()
for backbone in backbones:
    mod = importlib.import_module('backbones.' + backbone)
    class_name = {x.lower():x for x in mod.__dir__()}[backbone.replace('_', '')]
    names[backbone] = getattr(mod, class_name)

def get_backbone(backbone_name, indim, hiddim, outdim, args, configs):
    """Get the network architectures for encoder, predictor, discriminator."""
    model_name = names[backbone_name]
    print(model_name)
    model = model_name(indim, hiddim, outdim, args, configs)

    return model
