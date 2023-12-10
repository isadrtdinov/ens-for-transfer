import sys
import numpy as np

sys.path.append('.')
from main import main
from configs.resnet.resnet_starsse import Params


NUM_RUNS = 2
PT_CKPTS = [
    './checkpoints/ckpt_1.pth.tar',
    './checkpoints/ckpt_2.pth.tar',
]
LR_RATIOS = [0.25, 0.5, 1., 2., 4.]
EPOCH_RATIOS = [0.5, 1, 2, 4]

seeds = np.random.choice(100000, size=[len(PT_CKPTS), NUM_RUNS,
                                       len(LR_RATIOS), len(EPOCH_RATIOS)], replace=False)
for i, ckpt in enumerate(PT_CKPTS):
    for j in range(NUM_RUNS):
        for k, lr_ratio in enumerate(LR_RATIOS):
            for l, epoch_ratio in enumerate(EPOCH_RATIOS):
                config = Params(seed=seeds[i, j, k, l], ckpt=ckpt)
                config.fge_lr = lr_ratio * config.lr
                config.fge_epochs = int(epoch_ratio * config.epochs)
                main(config)
