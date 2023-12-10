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
<img align="right" width=40% src="https://github.com/isadrtdinov/ens-for-transfer/blob/master/images/logo.jpg" />
Transfer learning and ensembling are two popular techniques for improving the performance and robustness of neural networks. Due to the high cost of pre-training, ensembles of models fine-tuned from a single pre-trained checkpoint are often used in practice. Such models end up in the same basin of the loss landscape, which we call the pre-train basin, and thus have limited diversity. In this work, we show that ensembles trained from a single pre-trained checkpoint may be improved by better exploring the pre-train basin, however, leaving the basin results in losing the benefits of transfer learning and in degradation of the ensemble quality. Based on the analysis of existing exploration methods, we propose a more effective modification of the Snapshot Ensembles (SSE) for transfer learning setup, StarSSE, which results in stronger ensembles and uniform model soups.
</div>

## Code

### Environment
The project requirements are listed in `requirements.txt` file. To create a pip/conda environment:

```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

GPU with at least 32 Gb memory is required, e.g., NVIDIA V100/A100.

## Citation

If you found this code useful, please cite our paper

```
@inproceedings{sadrtdinov2023to,
    title={To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning},
    author={Ildus Sadrtdinov and Dmitrii Pozdeev and Dmitry P. Vetrov and Ekaterina Lobacheva},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
    year={2023},
}
```
