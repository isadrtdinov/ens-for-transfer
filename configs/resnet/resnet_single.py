from dataclasses import dataclass
from configs.resnet.resnet_base import ResNetBaseConfig


@dataclass
class Params(ResNetBaseConfig):
    dataset: str = 'cifar100'
    epochs: int = 25
    lr: float = 0.005
    wd: float = 5e-4
    wandb_group: str = 'ResNet_Baseline'
