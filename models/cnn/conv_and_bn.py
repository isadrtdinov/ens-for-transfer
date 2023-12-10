import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
copied from https://github.com/apple/learning-subspaces/blob/master/
experiment_configs/tinyimagenet/one_dimensional_subspaces/eval_lines.py
"""


def bezier(alpha: float, weights: nn.ParameterList):
    assert 0 <= alpha <= 1
    n = len(weights) - 1
    weight = sum(
        math.comb(n, i) * ((1 - alpha) ** (n - i)) * (alpha ** i) * weight_i
        for i, weight_i in enumerate(weights)
    )
    return weight


def simplex(alpha: np.array, weights: nn.ParameterList):
    assert np.all((alpha >= 0) & (alpha <= 1)) and np.allclose(alpha.sum(), 1)
    assert len(alpha) == len(weights)
    weight = sum(alpha_i * weight_i for alpha_i, weight_i in zip(alpha, weights))
    return weight


def get_weight(subspace_type, alpha, weight, bias):
    w, b = None, None
    if subspace_type == 'simplex':
        w = simplex(alpha, weight)
        if bias is not None:
            b = simplex(alpha, bias)

    elif subspace_type == 'bezier':
        w = bezier(alpha, weight)
        if bias is not None:
            b = bezier(alpha, bias)

    else:
        raise ValueError(f'Unknown subspace type: {subspace_type}')

    return w, b


class StandardConv(nn.Conv2d):
    def __init__(self, subspace_type=None, num_points=None, **kwargs):
        super().__init__(**kwargs)


class StandardBN(nn.BatchNorm2d):
    def __init__(self, subspace_type=None, num_points=None, **kwargs):
        super().__init__(**kwargs)


class StandardLinear(nn.Linear):
    def __init__(self, subspace_type=None, num_points=None, **kwargs):
        super().__init__(**kwargs)


class SubspaceConv(nn.Conv2d):
    def __init__(self, subspace_type='simplex', num_points=1, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.subspace_type = subspace_type
        if subspace_type == 'simplex':
            self.alpha = np.ones(num_points) / num_points
        elif subspace_type == 'bezier':
            self.alpha = 0.5
        else:
            raise ValueError(f'Unknown subspace type: {subspace_type}')

        self.weights = nn.ParameterList([nn.Parameter(torch.zeros_like(self.weight)) for _ in range(num_points)])
        for weight in self.weights:
            nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')
        self.weight.data = torch.Tensor()

        self.biases = None
        if getattr(self, 'bias', None) is not None:
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros_like(self.bias)) for _ in range(num_points)])
            self.bias.data = torch.Tensor()

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = get_weight(self.subspace_type, self.alpha, self.weights, self.biases)
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups,
        )
        return x


class SubspaceBN(nn.BatchNorm2d):
    def __init__(self, subspace_type='simplex', num_points=1, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.subspace_type = subspace_type
        if subspace_type == 'simplex':
            self.alpha = np.ones(num_points) / num_points
        elif subspace_type == 'bezier':
            self.alpha = 0.5
        else:
            raise ValueError(f'Unknown subspace type: {subspace_type}')

        self.weights = nn.ParameterList([nn.Parameter(torch.ones_like(self.weight)) for _ in range(num_points)])
        self.weight.data = torch.Tensor()

        self.biases = None
        if getattr(self, 'bias', None) is not None:
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros_like(self.bias)) for _ in range(num_points)])
            self.bias.data = torch.Tensor()

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = get_weight(self.subspace_type, self.alpha, self.weights, self.biases)

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None
            )
        return F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class SubspaceLinear(nn.Linear):
    def __init__(self, subspace_type='simplex', num_points=1, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.subspace_type = subspace_type
        if subspace_type == 'simplex':
            self.alpha = np.ones(num_points) / num_points
        elif subspace_type == 'bezier':
            self.alpha = 0.5
        else:
            raise ValueError(f'Unknown subspace type: {subspace_type}')

        self.weights = nn.ParameterList([nn.Parameter(torch.zeros_like(self.weight)) for _ in range(num_points)])
        for weight in self.weights:
            nn.init.kaiming_normal_(weight)
        self.weight.data = torch.Tensor()

        self.biases = None
        if getattr(self, 'bias', None) is not None:
            self.biases = nn.ParameterList([nn.Parameter(torch.zeros_like(self.bias)) for _ in range(num_points)])
            self.bias.data = torch.Tensor()

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = get_weight(self.subspace_type, self.alpha, self.weights, self.biases)
        x = F.linear(x, w, bias=b)
        return x
