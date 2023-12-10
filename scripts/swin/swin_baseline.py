import sys
import numpy as np

sys.path.append('.')
from main import main
from configs.swin.swin_single import Params


NUM_RUNS = 5
PT_CKPTS = [
    './checkpoints/swin_1.pth',
    './checkpoints/swin_2.pth',
    './checkpoints/swin_3.pth',
    './checkpoints/swin_4.pth',
    './checkpoints/swin_5.pth',
]

seeds = np.random.choice(100000, size=[len(PT_CKPTS), NUM_RUNS], replace=False)
for i, ckpt in enumerate(PT_CKPTS):
    for j in range(NUM_RUNS):
        config = Params(seed=seeds[i, j], ckpt=ckpt)
        main(config)
