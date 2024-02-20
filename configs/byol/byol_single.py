from dataclasses import dataclass
from configs.byol.byol_base import BYOLBaseConfig


@dataclass
class Params(BYOLBaseConfig):
    dataset: str = 'cifar100'
    epochs: int = 25
    lr: float = 0.05
    wd: float = 1e-4
    wandb_group: str = 'BYOL_Baseline'
