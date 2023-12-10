import sys
import numpy as np

sys.path.append('.')
from main import main
from configs.byol.byol_single import Params


NUM_RUNS = 5
PT_CKPTS = [
    './checkpoints/byol-1000ep-1.pt',
    './checkpoints/byol-1000ep-2.pt',
    './checkpoints/byol-1000ep-3.pt',
    './checkpoints/byol-1000ep-4.pt',
    './checkpoints/byol-1000ep-5.pt',
]

seeds = np.random.choice(100000, size=[len(PT_CKPTS), NUM_RUNS], replace=False)
for i, ckpt in enumerate(PT_CKPTS):
    for j in range(NUM_RUNS):
        config = Params(seed=seeds[i, j], ckpt=ckpt)
        main(config)
