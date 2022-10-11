# EgoTaskQA

<p align="left">
    <a href='[https://arxiv.org/abs/2210.00722](http://arxiv.org/abs/2210.03929)'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='[https://blog-img-1302618638.cos.ap-beijing.myqcloud.com/uPic/ICRA23_GenDexGrasp.pdf](https://buzz-beater.github.io/assets/publications/2022_egotaskqa_nips/paper.pdf)'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://sites.google.com/view/egotaskqa'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

This repo contains code for our NeurIPS Datasets and Benchmarks 2022 paper:

[EgoTaskQA: Understanding Human Tasks in Egocentric Videos](https://buzz-beater.github.io/assets/publications/2022_egotaskqa_nips/paper.pdf)

[Baoxiong Jia](https://buzz-beater.github.io/), [Ting Lei](https://scholar.google.com/citations?user=Zk7Vxz0AAAAJ&hl=en), [Song-Chun Zhu](http://www.stat.ucla.edu/~sczhu/), [Siyuan Huang](https://siyuanhuang.com/)

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
generation, balancing, data split, and baseline experiments. For these two functionalities, please checkout the corresponding
sub-directory for code and instructions.

## Citation
If you find our paper and/or code helpful, please consider citing:
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
