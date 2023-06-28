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
