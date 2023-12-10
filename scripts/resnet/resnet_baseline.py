import sys
import numpy as np

sys.path.append('.')
from main import main
from configs.resnet.resnet_single import Params


NUM_RUNS = 5
PT_CKPTS = [
    './checkpoints/ckpt_1.pth.tar',
    './checkpoints/ckpt_2.pth.tar',
    './checkpoints/ckpt_3.pth.tar',
    './checkpoints/ckpt_4.pth.tar',
    './checkpoints/ckpt_5.pth.tar',
]

seeds = np.random.choice(100000, size=[len(PT_CKPTS), NUM_RUNS], replace=False)
for i, ckpt in enumerate(PT_CKPTS):
    for j in range(NUM_RUNS):
        config = Params(seed=seeds[i, j], ckpt=ckpt)
        main(config)
