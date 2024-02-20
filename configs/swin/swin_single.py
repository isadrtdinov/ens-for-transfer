from dataclasses import dataclass
from configs.swin.swin_base import SwinBaseConfig


@dataclass
class Params(SwinBaseConfig):
    dataset: str = 'cifar100'
    epochs: int = 102
    lr: float = 1e-4
    wd: float = 0.25
    wandb_group: str = 'Swin_Baseline'
