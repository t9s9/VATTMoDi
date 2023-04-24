from typing import Tuple, Dict

import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, RMSprop, Adagrad

from .helper import exclude_from_weight_decay


def get_optimizer(name: str,
                  params,
                  optimizer_kwargs: Dict,
                  wd_ignore_layer_types: Tuple = (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d,),
                  wd_ignore_parameter_names: Tuple = ('bias', 'logit_scale', 'loss_weight')):
    weight_decay = optimizer_kwargs.pop('weight_decay', 0.0)
    # exclude given layer types and parameter names for weight decay and also filters out all parameters with
    # requires_grad=False
    parameter_groups = exclude_from_weight_decay(params,
                                                 weight_decay=weight_decay,
                                                 ignore_layer_types=wd_ignore_layer_types,
                                                 ignore_parameter_names=wd_ignore_parameter_names)

    name = name.lower()
    if name == 'adam':
        return Adam(parameter_groups, **optimizer_kwargs)
    elif name == 'adamw':
        return AdamW(parameter_groups, **optimizer_kwargs)
    elif name == 'sgd':
        return SGD(parameter_groups, **optimizer_kwargs)
    elif name == 'adagrad':
        return Adagrad(parameter_groups, **optimizer_kwargs)
    elif name == 'rmsprop':
        return RMSprop(parameter_groups, **optimizer_kwargs)
    else:
        raise ValueError(f'Unknown optimizer {name}. Use one of "adam", "adamw", "sgd", "adagrad", "rmsprop".')
