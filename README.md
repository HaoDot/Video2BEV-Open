<div align="center">

# UniV-Baseline
### The Official Baseline for **UniV**: the First Large-Scale Video-Based University Geo-Localization Benchmark  
**Drone Video ↔ Satellite Image Cross-View Retrieval & Navigation**

[![arXiv](https://img.shields.io/badge/arXiv-2411.13610-b31b1b.svg)](https://arxiv.org/abs/2411.13610)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/HaoDot/Video2BEV-Open?style=social)](https://github.com/HaoDot/Video2BEV-Open)
[![Forks](https://img.shields.io/github/forks/HaoDot/Video2BEV-Open?style=social)](https://github.com/HaoDot/Video2BEV-Open)

![](https://www.zdzheng.xyz/resource-img/UniV.webp)

• Watch full results on Bilibili: [![Bilibili](https://img.shields.io/badge/Bilibili-Watch-red?logo=bilibili)](https://www.bilibili.com/video/BV1SMhPzpEo2/)

</div>

## 🔥 Highlights

- First **video-to-satellite** geo-localization benchmark with **real drone videos** (not image-only!)
- Two challenging tasks:  
  → **Task 1**: Video-based drone-view → satellite localization  
  → **Task 2**: Satellite-guided drone video navigation  
- Zero university overlap between train/val/test → extremely challenging domain gap
- Full training/evaluation code + pretrained weights released
- Strong baseline using **Video2BEV + Two-stage training**

> Paper: **[Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization](https://arxiv.org/abs/2411.13610)** (ICCV 2025)

---



## TODOs

- [x] ~~(Optional) Release the 2-fps BEVs for both training and evaluation~~
- [x] ~~Release the `requirements.txt`~~
- [x] ~~Release the UniV dataset~~
- [x] ~~Release the weight of the second stage~~
- [x] ~~Release the ***evaluation*** code for the second stage~~
- [x] ~~Release the **training** code for the second stage~~
- [x] ~~Release the weight of the first stage~~
- [x] ~~Release the ***evaluation*** code for the first stage~~
- [x] ~~Release the **training** code for the first stage~~

<h1 align="left"> Table of contents</h1>

- [Dataset Introduction](#About-Dataset)
- [Getting started](#Getting-started)
- [Dataset & Preparation](#Dataset-&-Preparation)
- [Training & Evaluation](#Training-&-Evaluation)
- [Weights](#Weights)
- [Citation](#Citation)

## About Dataset

![image-20250730152824047](./assets/image-20250730152824047.png)

### Download

[BaiduCloud](https://pan.baidu.com/s/1fTEN3E2V82tia0JKAoTkrw?pwd=4g47)|[Google Drive](https://drive.google.com/drive/folders/14aWwrWIGzUp22krMgeimQXDRBsUkAAkK?usp=drive_link)|

The dataset split is as follows: 
| Split  for the each subset | #data | #buildings | #universities|
| --------   | -----  | ----| ----|
|Training | 701 **vids** + 12364 imgs | 701 | 33 |
| Query_drone | 701 **vids** | 701 |  39 |
| Query_satellite | 701 imgs | 701 | 39|
| Query_ground | 2,579 imgs | 701 | 39|
| Gallery_drone | 951 **vids** | 951 | 39|
| Gallery_satellite | 951 imgs | 951 | 39|
| Gallery_ground | 2,921 imgs | 793  | 39|

More detailed file structure:

```bash
.
├── 30
│   ├── 10fps
│   │   ├── test
│   │   │   └── gallery_drone
│   │   └── train
│   │       └── drone
│   ├── 2fps
│   │   ├── test
│   │   │   └── gallery_drone
│   │   └── train
│   │       └── drone
│   └── 5fps
│       ├── test
│       │   └── gallery_drone
│       └── train
│           └── drone
├── 45
│   ├── 10fps
│   │   ├── test
│   │   │   └── gallery_drone
│   │   └── train
│   │       └── drone
│   ├── 2fps
│   │   ├── test
│   │   │   ├── gallery_drone
│   │   │   ├── gallery_satellite
│   │   │   └── gallery_street
│   │   └── train
│   │       ├── drone
│   │       ├── google
│   │       ├── satellite
│   │       └── street
│   └── 5fps
│       ├── test
│       │   └── gallery_drone
│       └── train
│           └── drone
├── dataset_split.json
└── organize_univ.py
```

We note that there are no overlaps between 33 univeristies of training set and 39 univeristies of test set.

## Getting started

### Installation

```bash
conda create --name video2bev python=3.7
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# (optional but recommended) install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

If you have any question of installing apex, please refer to [issue-2](https://github.com/HaoDot/Video2BEV-Open/issues/2) first, then search for possible solutions.

## Dataset & Preparation
- Download UniV.
- cat and unzip the dataset: 
  - `mkdir UniV`
  - `cat UniV.tar.xz.* | tar -xJf - -C UniV --transform='s/\*/_/g'`
  - update the path of UniV in `organize_univ.py` and `python ./organize_univ.py`

- [Optional] If you are interested in reproducing or evaluating the proposed Video2BEV, please ~~feel free to contact us and ask for~~ download **BEVs** and **synthetic negative samples** (which is fine-tuned via [diffusers](https://github.com/huggingface/diffusers)) via [baidu-disk](https://pan.baidu.com/s/12fSIgKzrMtrs9JNCeYKOPA?pwd=1qve) or [google-drive](https://drive.google.com/drive/folders/1Yi8Ducbvb5LJW7stbeX6VUhJlKt-YDsb?usp=drive_link).
  - Unzip UniV-supp
    - `cd UniV-supp`
    - `cat UniV-supp.tar.xz.* | tar -xvJf - --transform 's|.*/|UniV-supp/|'`

  - Organize UniV-supp
    - set path in `organize_univ-supp.py` (in the above downloaded)
    - `python ./organize_univ-supp.py`

## Training & Evaluation

### Training

#### First-stage training & evaluation

- First-stage training:
  - Check to `first-stage` branch by `git checkout first-stage`
  - Refer to [this file](https://github.com/HaoDot/Video2BEV-Open/tree/first-stage)
- First-stage evaluation:
  - Check to `first-stage` branch by `git checkout first-stage`
  - Refer to [this file](https://github.com/HaoDot/Video2BEV-Open/tree/first-stage)

```bash
# Train:
# In the first stage, we fine-tune the encoder with the instance loss and contrastive loss.
sh train.sh
# Evaluation:
python test_collect_weights.py;
sh test.sh
```

####  Second-stage training & evaluation

- Second-stage training:
  - Check to `second-stage-training` branch by `git checkout second-stage-training`
  - Refer to [this file](https://github.com/HaoDot/Video2BEV-Open/tree/second-stage-training)
- Second-stage evaluation:
  - Check to `second-stage-evalution` branch by `git checkout second-stage-evalution`
  - Refer to [this file](https://github.com/HaoDot/Video2BEV-Open/blob/second-stage-evalution/README.md)

```bash
# Train:
# In the second stage, we freeze the encoder and train mlps with matching loss.
# please change contents in train.sh
sh train.sh

# Evaluation:
# please change contents in test_collect_weights.py and test.sh
python test_collect_weights.py;
sh test.sh
```

## Weights

[Download link](https://pan.baidu.com/s/1ZjssipR0RGfoPaETo4QhsQ?pwd=ahi4)

```bash
.
├── first-stage
│   ├── 30-degree
│   │   └── model_xxxx_xxxx
│   │       └── two_view_long_share_d0.75_256_s1
│   │           └── model_xxxx_xxxx_xxx
│   │               ├── net_9301.pth
│   │               └── opts.yaml
│   └── 45-degree
│       └── model_2024-08-20-19_19_36
│           └── two_view_long_share_d0.75_256_s1
│               └── model_2024-08-20-19_19_36_059
│                   ├── net_059.pth
│                   └── opts.yaml
├── second-stage
│   ├── 30degree-2fps
│   │   └── model_2024-11-02-03-05-31.zip
│   ├── 45degree-2fps
│   │   └── model_2024-10-05-02_49_11.zip
│   └── 45degree-2fps-better
│       └── model_2024-10-20-06_02_09.zip
└── vit_small_p16_224-15ec54c9.pth
```

Choose the weight and **unzip** it. Then put it in the root path in the working directory for your repo.

PS: 

- `model_2024-11-02-03-05-31` is the weight for 30-degree UniV (2fps) and `model_2024-10-05-02_49_11` is the weight for 45-degree UniV (2fps)
  - The evaluation number should be the same as our paper
- By tuning hyper-parameter, we can get a better result.



------

## Citation

The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```bibtex
@article{ju2024video2bev,
  title={Video2bev: Transforming drone videos to bevs for video-based geo-localization},
  author={Ju, Hao and Huang, Shaofei and Liu, Si and Zheng, Zhedong},
  journal=ICCV,
  year={2025}
}
```
Others:

```bibtex
@article{zheng2020university,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  journal={ACM Multimedia},
  year={2020}
}
@article{zheng2017dual,
  title={Dual-Path Convolutional Image-Text Embeddings with Instance Loss},
  author={Zheng, Zhedong and Zheng, Liang and Garrett, Michael and Yang, Yi and Xu, Mingliang and Shen, Yi-Dong},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  doi={10.1145/3383184},
  volume={16},
  number={2},
  pages={1--23},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```

