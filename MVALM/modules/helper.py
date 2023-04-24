from typing import Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def contrastive_accuracy(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    num_logits = feat_a.shape[0]
    ground_truth = torch.arange(num_logits, device=feat_a.device, dtype=torch.long)

    logits_per_audio = feat_a @ feat_b.T
    acc_per_a = (torch.argmax(logits_per_audio, 1) == ground_truth).sum().item()
    acc_per_b = (torch.argmax(logits_per_audio, 0) == ground_truth).sum().item()
    return (acc_per_a + acc_per_b) / 2 / num_logits


def get_parameter_names(model: nn.Module, ignore_layer_types: Tuple, ignore_parameter_names: Tuple):
    """
    Recursively parses the model to find all layer names that are not in `ignore_layer_type` and do not contain
    `ignore_parameter_names`
    """
    layers = []
    for name, child in model.named_children():
        if not isinstance(child, ignore_layer_types):
            for layer in get_parameter_names(child, ignore_layer_types, ignore_parameter_names):
                parameter_name = f'{name}.{layer}'
                if not any(x in parameter_name for x in ignore_parameter_names):
                    layers.append(parameter_name)

    for parameter_name in list(model._parameters.keys()):
        if not any(x in parameter_name for x in ignore_parameter_names):
            layers.append(parameter_name)

    return layers


def exclude_from_weight_decay(model: nn.Module,
                              weight_decay: float,
                              ignore_layer_types: Tuple,
                              ignore_parameter_names: Tuple):
    # Don't apply weight decay to bias and Normalization layers
    # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/8
    # https://arxiv.org/pdf/1706.05350.pdf
    params = []
    excluded_params = []

    apply_wd_layers = get_parameter_names(model, ignore_layer_types, ignore_parameter_names)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif name in apply_wd_layers:
            params.append(param)
        else:
            excluded_params.append(param)

    result = []
    if len(params) > 0:
        result.append({"params": params, "weight_decay": weight_decay})
    if len(excluded_params) > 0:
        result.append({"params": excluded_params, "weight_decay": 0.0})
    if not result:
        raise ValueError("No parameters to optimize.")
    return result
