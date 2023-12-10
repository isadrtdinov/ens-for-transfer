from typing import Union, List, Dict, cast
from collections import defaultdict
import torch
import torch.nn as nn
import math
from .builder import Builder

__all__ = [
    "VGG",
    "VGG16",
    "VGG16BN",
    "VGG19",
    "VGG19BN",
]

cfgs: Dict[int, List[Union[str, int]]] = {
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(builder, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = builder.conv3x3(in_channels, v)
            if batch_norm:
                layers += [builder.conv3x3(in_channels, v, bias=False), builder.batchnorm(v), nn.ReLU(inplace=True)]
            else:
                layers += [builder.conv3x3(in_channels, v, bias=True), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(
        self, depth: str, batch_norm: bool = False, num_classes: int = 1000, 
        dropout: float = 0.5, builder_kwargs={}
    ) -> None:
        super().__init__()

        builder = Builder(**builder_kwargs)
        self.features = make_layers(builder, cfgs[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            builder.linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            builder.linear(512, 512),
            nn.ReLU(inplace=True),
            builder.linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if getattr(m, 'bias', None) is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_layerwise_dict(self):
        result_dict = defaultdict(list, {'full': []})
        cur_layer = 1

        last_module_n = list(self.features.named_modules())[-1][0]
        for module_n in range(last_module_n + 1):
            layer_name = 'layer' + str(cur_layer)
            module_name = 'feature.' + str(module_n)
            
            result_dict[layer_name].append(module_name)
            if isinstance(self.features.get_submodule(module_name), nn.MaxPool2d):
                cur_layer += 1


class VGG16:
    base = VGG
    kwargs = {
        'depth': 16,
        'batch_norm': False
    }


class VGG16BN:
    base = VGG
    kwargs = {
        'depth': 16,
        'batch_norm': True
    }


class VGG19:
    base = VGG
    kwargs = {
        'depth': 19,
        'batch_norm': False
    }


class VGG19BN:
    base = VGG
    kwargs = {
        'depth': 19,
        'batch_norm': True
    }
