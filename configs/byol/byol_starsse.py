from dataclasses import dataclass
from configs.ssl.byol.byol_base import BYOLBaseConfig


@dataclass
class Params(BYOLBaseConfig):
    dataset: str = 'cifar100'
    scheduler: str = 'cosine+warmup_cosine'
    epochs: int = 3
    lr: float = 0.05
    wd: float = 1e-4
    wandb_group: str = 'BYOL_StarSSE'

    num_fge: int = 2
    fge_epochs: int = 3
    fge_lr: float = 0.05
    fge_warmup_epochs: int = 0
    star_fge: bool = True
