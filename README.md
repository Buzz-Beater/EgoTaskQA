# EgoTaskQA

This repo contains code for our NeurIPS Datasets and Benchmarks 2022 paper:

[EgoTaskQA: Understanding Human Tasks in Egocentric Videos]

Baoxiong Jia, Ting Lei, Song-Chun Zhu, Siyuan Huang
*Advances in Neural Information Processing Systems Datasets and Benchmarks (NeurIPS Datasets and Benchmarks)*, 2022

## Dataset
For data download, please check our [website](https://sites.google.com/view/egotask-qa) for instructions and details.
![overview](https://buzz-beater.github.io/assets/publications/2022_egotaskqa_nips/overview.png)
## Experimental Setup
We provide all environment configurations in ``requirements.txt``. In our experiments, we used NVIDIA CUDA 11.3 on Ubuntu 20.04
and need this additional step for version control on pytorch:
```bash
$ conda create -n egotaskqa python=3.8
$ pip install -r requirements.txt
$ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113 
```

Similar CUDA version should also be acceptable with corresponding version control for ``torch`` and ``torchvision``.
We refer the authors to [Generation](generation/README.md) and [Experiment](baselines/README.md) for details on quetsion-answer
generation, balancing, data split, and baseline experiments.

## Citation
If you find our paper and/or code helpful, please consider citing
```
@inproceedings{jia2022egotaskqa,
    title = {EgoTaskQA: Understanding Human Tasks in Egocentric Videos},
    author = {Jia, Baoxiong and Lei, Ting and Zhu, Song-Chun and Huang, Siyuan},
    booktitle = {The 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
    year = {2022}
}
```

## Acknowledgement
We thank all colleagues from VCLA and BIGAI for fruitful discussions. We would also like to thank the anonymous reviewers for their constructive feedback.