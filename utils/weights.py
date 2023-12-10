import torch
import numpy as np
from collections import defaultdict
from torch import nn
import models


def get_simplex_grid(num_points):
    grid = []
    xs = np.linspace(0, 1, num_points)
    for i, x in enumerate(xs):
        y = np.linspace(0, 1 - x, num_points - i)
        x = np.full_like(y, x)
        grid.append(np.stack([x, y, 1 - x - y], axis=1))

    grid = np.concatenate(grid, axis=0)
    vertices_pos = (0, num_points - 1, grid.shape[0] - 1)
    return grid, vertices_pos


def set_alpha(model, alpha):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            setattr(m, f"alpha", alpha)


def split_model(subspace_model, num_points, args):
    split_models = []
    for i in range(num_points):
        architecture = getattr(models, args.model)
        model = architecture.base(num_classes=args.num_classes, **architecture.kwargs)

        subspace_state_dict = subspace_model.state_dict()
        for name, param in model.named_parameters():
            subspace_name = name.replace('weight', 'weights').replace('bias', 'biases') + f'.{i}'
            param.data.copy_(subspace_state_dict[subspace_name])

        split_models.append(model.to(args.device))

    return split_models


def transfer_weights(model, subspace_model):
    model_state_dict = model.state_dict()

    for name, param in subspace_model.named_parameters():
        if param.numel() == 0:
            continue

        model_name = name[:name.rfind('.')].replace('weights', 'weight').replace('biases', 'bias')
        param.data.copy_(model_state_dict[model_name])


def add_subspace_point(model, freeze_grad=True):
    def add_weight(param_list):
        init_weight = torch.stack([weight.data.clone() for weight in param_list], dim=0).mean(dim=0)
        param_list.append(nn.Parameter(init_weight))

        if freeze_grad:
            for i in range(len(param_list) - 1):
                param_list[i].requires_grad = False

    device = next(model.parameters()).device
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            m.num_points += 1
            add_weight(m.weights)

            if hasattr(m, 'biases') and m.biases is not None:
                add_weight(m.biases)

    model.to(device)


def get_weights(model):
    params = [p.data.cpu().numpy().ravel() for p in model.parameters()]
    return np.concatenate(params) if len(params) else []


def weights_to_model(model, w):
    offset = 0
    for parameter in model.parameters():
        size = np.prod(parameter.size())
        value = w[offset:offset+size].reshape(parameter.size())
        parameter.data.copy_(torch.from_numpy(value))
        offset += size

    return model


def add_noise_to_weights(model, noise_ratio=0.01):
    weights = get_weights(model)
    noise = np.random.randn(weights.shape[0])
    noise *= noise_ratio * np.linalg.norm(weights) / np.linalg.norm(noise)
    weights_to_model(model, weights + noise)


def get_conv_and_weights_layerwise(model, model_name=None):
    def get_conv_weights(model):
        cur_weights = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                cur_weights += [p.data.numpy().ravel() for p in m.parameters()]

        return np.concatenate(cur_weights) if cur_weights else []

    backbone_name = 'backbone'
    names_dict = {backbone_name: []}

    assert model_name is not None
    if 'resnet' in model_name.lower() or 'cifarresnet' in model_name.lower():
        classifier_name = 'fc'
        names_dict.update(
            {'conv1': [], 'bn1': [], 'layer1': [], 'layer2': [], 'layer3': [], 'layer4': [], 'fc': []}
        )
    elif 'vgg16' in model_name.lower():
        classifier_name = 'classifier'
        cur_dict = defaultdict(list)
        cur_layer = 1

        last_module_n = int(list(model.features.named_modules())[-1][0])
        for module_n in range(last_module_n + 1):
            layer_name = 'layer' + str(cur_layer)
            module_name = 'features.' + str(module_n)

            cur_dict[layer_name].append(module_name)
            if isinstance(model.get_submodule(module_name), nn.MaxPool2d):
                cur_layer += 1

        names_dict.update(cur_dict)
    else:
        raise ValueError("Unknown type of model for getting layer wise statistics")

    results = {'conv': {}, 'all': {}}

    for layer_name, module_names in names_dict.items():
        if layer_name == backbone_name:
            classifier = getattr(model, classifier_name)
            setattr(model, classifier_name, nn.Identity())
            results['conv'][layer_name] = get_conv_weights(model)
            results['all'][layer_name] = get_weights(model)
            setattr(model, classifier_name, classifier)
        elif not module_names:
            results['conv'][layer_name] = get_conv_weights(getattr(model, layer_name))
            results['all'][layer_name] = get_weights(getattr(model, layer_name))
        else:
            all_layer_weights = []
            conv_layer_weights = []
            for name in module_names:
                conv_layer_weights += [get_conv_weights(model.get_submodule(name))]
                all_layer_weights += [get_weights(model.get_submodule(name))]

            results['conv'][layer_name] = np.concatenate(conv_layer_weights)
            results['all'][layer_name] = np.concatenate(all_layer_weights)

    return results
