
# Adversarial Attack Evaluation
This repository contains the code for [	
Reliable Evaluation of Adversarial Transferability
](http://arxiv.org/abs/2306.08565)

## - Requirements
- python 3.7
- pytorch 1.13.1
- torchvision 0.14.1

## - Pretrained models
- SNNs: please first download the checkpoints from [Google Drive](https://drive.google.com/drive/folders/1vwNx4xTF6EG_Brbu-6mGkgC2HcfgtBTe) and put the folder under "codes/model/snn_models/" directory.
- GFNets: please first download the checkpoints from [Res50/96](https://drive.google.com/file/d/1Iun8o4o7cQL-7vSwKyNfefOgwb9-o9kD/view?usp=sharing), [Res50/128](https://drive.google.com/file/d/1cEj0dXO7BfzQNd5fcYZOQekoAe3_DPia/view?usp=sharing), [D121/96](https://drive.google.com/file/d/1UflIM29Npas0rTQSxPqwAT6zHbFkQq6R/view?usp=sharing) and put them under "codes/model/gfnet/checkpoints/" directory. 
## - How to use

Four types of model can be tested：CNNs, ViTs, SNNs and GFNets.

In our experiments, 12 attack methods are evaluated and all of them are based on PGD attack. PGD, TI, DI, MI, NI and VMI are applied to all source models; LinBP can only be used on ResNet50; SGM is applied to two source models (ResNet50 and DenseNet121); ILA is also applied to two source mdoels (DenseNet121 and IncepeionV3); IR is used on four CNN models; Self-Ensemble (SE) and Pay No Attention (PNA) are ViT-specific attacks.

Taken VGG16 as an example: 

- generate adversarial examples：

```
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD' --generate_mode
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD+TI' --generate_mode --ti_size=15
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD+DI' --generate_mode --prob=0.4
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD+MI' --generate_mode --momentum=1.0
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD+NI' --generate_mode --momentum=1.0
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD+MI+VI' --generate_mode --momentum=1.0
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='vgg16' --tar_kind='cnn' --attack_method='PGD+IR' --generate_mode  --lam=1.0
```

- evaluate
```
python main_interaction_loss.py --src_model='vgg16' --src_kind='cnn' --tar_model='ensemble' --tar_kind='ensemble' --attack_method='PGD'
```

## - Results
  
| Model | VGG16 | Res50 | Dense121 | IncV3 | ViT-B/16 | Deit-B/16 | Swin-B/4/7 | Avg |
| :----: | :----: | :----: | :----: | :----: | :---: | :---: | :------: | :------: |
| PGD | 4.74/<br>5.49/<br>0.06 | 2.68/<br>4.22/<br>0.10 | 5.52/<br>5.19/<br>0.16 | 0.62/<br>2.42/<br>0.04 | 7.22/<br>5.11/<br>0.40 | 4.54/<br>5.31/<br>0.26 | 0.76/<br>2.66/<br>0.06 | 3.73/<br>4.34/<br>0.15 |
| TI | 7.54/<br>5.73/<br>0.12 | 4.86/<br>4.81/<br>0.14 | 7.62/<br>5.27/<br>0.22 | 1.80/<br>3.60/<br>0.08 | 6.64/<br>4.54/<br>0.22 | 3.92/<br>4.43/<br>0.42 | 3.72/<br>3.99/<br>0.34 | 5.16/<br>4.62/<br>0.22 |
| DI | 9.88/<br>6.59/<br>0.34 | 14.24/<br>7.01/<br>0.68 | 20.10/<br>7.82/<br>0.82 | 2.20/<br>3.73/<br>0.04 | 36.62/<br>9.40/<br>5.56 | 27.74/<br>9.49/<br>5.28 | 10.40/<br>5.45/<br>1.08 | 17.31/<br>7.07/<br>1.97 |
| SGM | /<br>| 21.62/<br>7.74/<br>1.22 | 16.22/<br>6.95/<br>0.98 | /<br>| /<br>| /<br>| /<br>| 18.92/<br>7.345/<br>1.71 |
| LinBP | /<br>| 13.44/<br>8.29/<br>0.52 | /<br>| /<br>| /<br>| /<br>| /<br>| 13.44/<br>8.29/<br>0.52 |
| ILA | /<br>| /<br>| 7.64/<br>6.66/<br>0.38 | 1.98/<br>5.52/<br>0.04 | /<br>| /<br>| /<br>| 4.81/<br>6.09/<br>0.21 |
| IR | 11.66/<br>6.62/<br>0.52 | 11.16/<br>6.27/<br>0.54 | 15.64/<br>7.35/<br>0.84 | 2.52/<br>3.59/<br>0.12 | /<br>| /<br>| /<br>| 10.25/<br>5.96/<br>0.51 |
| MI | 10.36/<br>6.91/<br>0.34 | 8.60/<br>6.51/<br>0.38 | 14.50/<br>7.50/<br>0.78 | 2.28/<br>4.04/<br>0.10 | 17.80/<br>7.65/<br>1.52 | 14.88/<br>8.12/<br>1.34 | 3.96/<br>4.55/<br>0.30 | 10.34/<br>6.47/<br>0.68 |
| NI | 10.62/<br>6.92/<br>0.48 | 8.66/<br>6.53/<br>0.46 | 14.12/<br>7.49/<br>0.74 | 2.38/<br>4.01/<br>0.10 | 18.14/<br>7.71/<br>1.80 | 14.54/<br>8.11/<br>1.32 | 4.12/<br>4.57/<br>0.28 | 10.37/<br>6.48/<br>0.74 |
| VMI | 10.34/<br>6.84/<br>0.28 | 7.58/<br>6.10/<br>0.42 | 13.20/<br>7.15/<br>0.60 | 3.38/<br>0.06/<br>1.40 | 9.40/<br>5.87/<br>0.60 | 7.54/<br>5.92/<br>0.54 | 3.86/<br>4.47/<br>0.32 | 7.9/<br>5.20/<br>0.59 |

## - Citation

Please cite the following paper, if you use this code.

```
@article{yu2023reliable,
  title={Reliable Evaluation of Adversarial Transferability},
  author={Yu, Wenqian and Gu, Jindong and Li, Zhijiang and Torr, Philip},
  journal={arXiv preprint arXiv:2306.08565},
  year={2023}
}
```
