from . import conv_and_bn


# modified from https://github.com/apple/learning-subspaces/blob/master/models/builder.py
# class for conv3x3 initialization and return


class Builder(object):
    def __init__(self, subspace_type=None, num_points=None):
        if subspace_type is not None:
            assert num_points is not None

        self.subspace_type = subspace_type
        self.num_points = num_points

        self.conv_layer = conv_and_bn.StandardConv if subspace_type is None else conv_and_bn.SubspaceConv
        self.bn_layer = conv_and_bn.StandardBN if subspace_type is None else conv_and_bn.SubspaceBN
        self.linear_layer = conv_and_bn.StandardLinear if subspace_type is None else conv_and_bn.SubspaceLinear

    def conv(self, in_planes, out_planes, kernel_size, stride: int = 1, padding: int = 0,
             groups: int = 1, dilation: int = 1, bias: bool = True,
    ):
        """general convolution"""
        return self.conv_layer(
            self.subspace_type, self.num_points,
            in_channels=in_planes, out_channels=out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, groups=groups,
            dilation=dilation, bias=bias
        )

    def conv1x1(self, in_planes: int, out_planes: int, stride: int = 1, bias=False):
        """1x1 convolution"""
        return self.conv_layer(
            self.subspace_type, self.num_points,
            in_channels=in_planes, out_channels=out_planes,
            kernel_size=1, stride=stride, bias=bias
        )

    def conv3x3(self, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias=False):
        """3x3 convolution with padding"""
        return self.conv_layer(
            self.subspace_type, self.num_points,
            in_channels=in_planes, out_channels=out_planes,
            kernel_size=3, stride=stride,
            padding=dilation, groups=groups,
            bias=bias, dilation=dilation,
        )

    def batchnorm(self, num_features):
        bn = self.bn_layer(self.subspace_type, self.num_points, num_features=num_features)
        return bn

    def linear(self, in_features, out_features):
        linear = self.linear_layer(self.subspace_type, self.num_points,
                                   in_features=in_features, out_features=out_features)
        return linear
