import os
import numpy as np
from dataclasses import dataclass


@dataclass
class SwinAugConfig:
    color_jitter: float = 0.4
    # use autoaugment policy. "v0" or "original"
    auto_augment: str = 'rand-m9-mstd0.5-inc1'
    # random erase prob
    reprob: float = 0.25
    # random erase mode
    remode: str = 'pixel'
    # random erase count
    recount: int = 1
    # mixup alpha, mixup enabled if > 0
    mixup: float = 0.8
    # cutmix alpha, cutmix enabled if > 0
    cutmix: float = 1.0
    # cutmix min/max ratio, overrides alpha and enables cutmix if set
    cutmix_minmax: float = None
    # probability of performing mixup or cutmix when either/both is enabled
    mixup_prob: float = 1.0
    # probability of switching to cutmix when both mixup and cutmix enabled
    mixup_switch_prob: float = 0.5
    # how to apply mixup/cutmix params. per "batch", "pair", or "elem"
    mixup_mode: str = 'batch'


@dataclass
class SwinModelConfig:
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    qk_scale: float = None
    ape: bool = False
    patch_norm: bool = True


@dataclass
class SwinBaseConfig:
    # Swin params
    aug: SwinAugConfig = SwinAugConfig()
    swin: SwinModelConfig = SwinModelConfig()

    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    label_smoothing: float = 0.1
    layer_decay: float = 0.85  # For SimMIM only, value from UM-MAE

    fused_layernorm: bool = False
    fused_window_process: bool = False
    use_checkpointing: bool = False
    simmim: bool = False

    # Seed
    seed: int = None

    # Checkpoint and directories params
    work_dir: str = './experiments/'
    save_path: str = None
    ckpt: str = './checkpoints/swin_1.pth'
    save_logits: bool = True

    # Model, dataset and augmentation params
    dataset: str = 'cifar100'
    data_path: str = './datasets/'
    model: str = 'SwinTiny'
    transform_name: str = 'swin_imagenet'
    interpolation: str = 'bicubic'
    use_test: bool = True
    use_mixup: bool = True
    drop_last: bool = True
    num_workers: int = 8

    # Training params
    epochs: int = 1
    start_epoch: int = 1
    batch_size: int = 256

    # Optimizer params
    optimizer: str = 'Adamw'
    adam_eps: float = 1e-8
    adam_betas: tuple = (0.9, 0.999)
    lr: float = 5e-3
    wd: float = 0.05
    warmup_lr: float = 5e-7
    scheduler: str = 'linear_warmup_cosine'
    warmup_epochs: int = None
    warmup_coef: float = 0.1

    # Optimization utils params
    loss_scaler: str = 'amp_scaler'
    amp_enable: bool = True
    clip_grad: float = 5.0

    # FGE/SSE params
    num_fge: int = 1
    fge_epochs: int = None
    fge_lr: float = None
    fge_warmup_epochs: int = None
    star_fge: bool = False
    reset_optim_state: bool = True

    # Logging params
    valid_freq: int = 4
    wandb_log_rate: int = 30
    wandb_project: str = 'EnsForTransfer'
    wandb_group: str = 'Swin_Test'
    exp_name: str = ''

    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(1000000)

        self.work_dir = os.path.join(
            self.work_dir, self.dataset, self.wandb_group,
            self.model.lower() + '_seed=' + str(self.seed) + '/'
        )

        self.save_path = \
            f'{self.work_dir}model={self.model}_dataset_={self.dataset}_sched={self.scheduler}_' \
            f'lr={self.lr}_wd={self.wd}_epoch={self.epochs}.pt'

        if self.warmup_epochs is None:
            self.warmup_epochs = int(self.epochs * self.warmup_coef)
