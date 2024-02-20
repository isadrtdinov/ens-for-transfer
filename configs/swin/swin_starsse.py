from dataclasses import dataclass
from configs.swin.swin_base import SwinBaseConfig


@dataclass
class Params(SwinBaseConfig):
    dataset: str = 'cifar100'
    scheduler: str = 'linear_warmup_cosine+cosine'
    epochs: int = 102
    lr: float = 1e-4
    wd: float = 0.25
    wandb_group: str = 'Swin_Baseline'

    num_fge: int = 5
    fge_epochs: int = 102
    fge_lr: float = 1e-4
    fge_warmup_epochs: int = 0
    star_fge: bool = True
