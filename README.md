# To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning

This repo is the official source code of NeurIPS-2023 paper:

**To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning** \
By [Ildus Sadrtdinov](https://scholar.google.com/citations?user=XhqNegUAAAAJ&hl=en)\*,
[Dmitrii Pozdeev](https://scholar.google.com/citations?user=4dlh3pkAAAAJ&hl=en)\*,
[Dmitry Vetrov](https://scholar.google.com/citations?user=7HU0UoUAAAAJ&hl=en),
[Ekaterina Lobacheva](https://tipt0p.github.io/)

[arXiv](https://arxiv.org/abs/2303.03374) / [OpenReview](https://openreview.net/forum?id=NNooZoQpP4&noteId=9lHQopv0ZL) / [Poster & video](https://nips.cc/virtual/2023/poster/71864)

## Abstract

<div align="justify">
<img align="right" width=35% src="https://github.com/isadrtdinov/ens-for-transfer/blob/master/images/logo.jpg" />
Transfer learning and ensembling are two popular techniques for improving the performance and robustness of neural networks. Due to the high cost of pre-training, ensembles of models fine-tuned from a single pre-trained checkpoint are often used in practice. Such models end up in the same basin of the loss landscape, which we call the pre-train basin, and thus have limited diversity. In this work, we show that ensembles trained from a single pre-trained checkpoint may be improved by better exploring the pre-train basin, however, leaving the basin results in losing the benefits of transfer learning and in degradation of the ensemble quality. Based on the analysis of existing exploration methods, we propose a more effective modification of the Snapshot Ensembles (SSE) for transfer learning setup, StarSSE, which results in stronger ensembles and uniform model soups.
</div>

## Code

### Environment
The project requirements are listed in `requirements.txt` file. To create a pip/conda environment:

```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name ens_for_transfer --file requirements.txt
```

Note:
- As during fine-tuning the images are resized to `224x224` and given `batch_size=256`, training requires a GPU with at least **32 Gb memory**, e.g., **NVIDIA V100/A100**.
- Logging is done with the [`wandb`](https://wandb.ai/) library, so make sure to log in before launching the experiments.

### Configuration and training parameters

To see the configuration files and parameters for different training setups, refer to the [`configs/`](https://github.com/isadrtdinov/ens-for-transfer/tree/master/configs) directory.

### Pre-trained checkpoints

- BYOL ResNet-50 ImageNet pre-trained checkpoints are available [here](https://drive.google.com/drive/folders/1BONZZ6pytC3yP2EXcZJaB07z4eKmtx20?usp=sharing)
- Supervised ResNet-50 ImageNet checkpoints pre-trained by [Asukha et al, 2020](https://github.com/SamsungLabs/pytorch-ensembles)
- Supervised Swin-Tiny ImageNet pre-trained checkpoints are available [here](https://drive.google.com/drive/folders/1gF_T3r7cyDO-JqnIGnWUVQy2GvSHR_IC?usp=sharing)

### Experiments

Scripts for launching experiments are located in the [`scripts/`](https://github.com/isadrtdinov/ens-for-transfer/tree/master/scripts) directory. For example, to launch experiments for BYOL ResNet-50 setup, run the following commands:

- For training baselines (Local DE and Global DE)
```sh
python scripts/byol/byol_baseline.py
```

- For training SSE with different cycle hyperparameters
```sh
python scripts/byol/byol_sse.py
```
  
- For training StarSSE with different cycle hyperparameters
```sh
python scripts/byol/byol_starsse.py
```

## Citation

If you found this code useful, please cite our paper:

```
@inproceedings{sadrtdinov2023to,
    title={To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning},
    author={Ildus Sadrtdinov and Dmitrii Pozdeev and Dmitry P. Vetrov and Ekaterina Lobacheva},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
    year={2023},
}
```
