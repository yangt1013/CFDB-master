# Coarse-to-fine dual-branch network for ship target recognition in complex environments

## Requirement
python 3.8

Pytorch >=1.7

torchvision >=0.8

## Training

1. Download datatsets for GLPM (e.g. MAR-ships, CIB-ships, game-of-ships etc) and organize the structure as follows:
```bash
dataset

└── train/test

    ├── class_001
    
    |      ├── 1.jpg    
    |      ├── 2.jp
    |      └── ...    
    ├── class_002
    
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```
```
2、Train from scratch with `train.py`.
## MAR-ships dataset link:
```bash
ARGOS-Venice boat classification

website：https://pan.baidu.com/s/1OHBMLMXvkKima1nK5gF-4A?pwd=GLPM 
word：GLPM 
```
## game-of-ships dataset link:
```bash
website：https://pan.baidu.com/s/1XkSwtPnxKblxZ6YQV4tEDA?pwd=GLPM 
word：GLPM 
```
