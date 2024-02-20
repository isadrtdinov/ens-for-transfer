from dataclasses import dataclass
from configs.byol.byol_base import BYOLBaseConfig


@dataclass
class Params(BYOLBaseConfig):
    dataset: str = 'cifar100'
    scheduler: str = 'cosine+warmup_cosine'
    epochs: int = 25
    lr: float = 0.05
    wd: float = 1e-4
    wandb_group: str = 'BYOL_SSE'

    num_fge: int = 5
    fge_epochs: int = 25
    fge_lr: float = 0.05
    fge_warmup_epochs: int = 0
    star_fge: bool = False
