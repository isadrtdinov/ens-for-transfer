import math
import random
import torch
from torch import nn
from abc import ABC


class Regularization(nn.Module, ABC):
    def __init__(self, num_points=None):
        super().__init__()
        self.num_points = num_points
        self.eps = 1e-4

    @staticmethod
    def get_conv_weights(model, index=None):
        weights = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if index is None:
                    # standard model
                    weight = m.weight
                else:
                    # subspace model
                    weight = m.weights[index]
                weights += [weight.reshape(-1)]

        return torch.cat(weights)


class CosineRegularization(Regularization):
    """
    adapted from https://github.com/apple/learning-subspaces/blob/
    9e4cdcf4cb92835f8e66d5ed13dc01efae548f67/trainers/train_one_dim_subspaces.py#L55
    """
    @staticmethod
    def cosine(x, y):
        sum_sq = (x * y).sum().square()
        norm_x_sq = x.square().sum()
        norm_y_sq = y.square().sum()
        return sum_sq / (norm_x_sq * norm_y_sq)

    def forward(self, model):
        if isinstance(model, nn.Module):
            # subspace learning model
            i, j = random.sample(range(self.num_points), 2)
            weights_i = self.get_conv_weights(model, index=i)
            weights_j = self.get_conv_weights(model, index=j)
        else:
            # assumed to be a list of models
            i, j = random.sample(range(len(model)), 2)
            weights_i = self.get_conv_weights(model[i])
            weights_j = self.get_conv_weights(model[j])

        return self.cosine(weights_i, weights_j)


class L2Regularization(Regularization):
    @staticmethod
    def l2(x, y):
        return (x - y).square().sum()

    def forward(self, model):
        if isinstance(model, nn.Module):
            # subspace learning model
            i, j = random.sample(range(self.num_points), 2)
            weights_i = self.get_conv_weights(model, index=i)
            weights_j = self.get_conv_weights(model, index=j)
        else:
            # assumed to be a list of models
            i, j = random.sample(range(len(model)), 2)
            weights_i = self.get_conv_weights(model[i])
            weights_j = self.get_conv_weights(model[j])

        return -torch.log(self.l2(weights_i, weights_j) + self.eps)


class VolumeRegularization(Regularization):
    """
    adapted from https://github.com/g-benton/loss-surface-simplexes/blob/
    0f55807ae16d6dbc8e41ec6ae59cc74884574cc0/simplex/models/basic_simplex.py#L84
    """
    def volume(self, vertices):
        dists = torch.square(
            vertices.reshape(1, vertices.shape[0], -1) -
            vertices.reshape(vertices.shape[0], 1, -1)
        ).sum(dim=-1)

        matrix = torch.ones(self.num_points + 1, self.num_points + 1, device=vertices.device) - \
            torch.eye(self.num_points + 1, device=vertices.device)
        matrix[:dists.shape[0], :dists.shape[0]] = dists
        norm = (math.factorial(self.num_points - 1) ** 2) * (2 ** (self.num_points - 1))

        return torch.abs(torch.det(matrix)) / norm

    def forward(self, model):
        if isinstance(model, nn.Module):
            # subspace learning model
            weights = [self.get_conv_weights(model, index=i) for i in range(self.num_points)]
        else:
            # assumed to be a list of models
            weights = [self.get_conv_weights(single_model) for single_model in model]

        weights = torch.stack(weights, dim=0)
        volume = self.volume(weights)
        return -torch.log(volume + self.eps)
