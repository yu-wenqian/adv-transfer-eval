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

            <table style="width:100%;float:left; font-size:80%;">
                <tr style="border-bottom: solid 1px black;border-top: solid 1px black;">
                    <td style="border-right: solid 1px black;">Model</td>
                    <td style="border-right: solid 1px black;">VGG16</td>
                    <td style="border-right: solid 1px black;">Res50</td>
                    <td style="border-right: solid 1px black;">Dense121</td>
                    <td style="border-right: solid 1px black;">IncV3</td>
                    <td style="border-right: solid 1px black;">ViT-B/16</td>
                    <td style="border-right: solid 1px black;">Deit-B/16</td>
                    <td style="border-right: solid 1px black;">Swin-B/4/7</td>
                    <td>Avg</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">PGD</td>
                    <td style="border-right: solid 1px black;">4.74/ 5.49/ 0.06</td>
                    <td style="border-right: solid 1px black;">2.68/ 4.22/ 0.10</td>
                    <td style="border-right: solid 1px black;">5.52/ 5.19/ 0.16</td>
                    <td style="border-right: solid 1px black;">0.62/ 2.42/ 0.04</td>
                    <td style="border-right: solid 1px black;">7.22/ 5.11/ 0.40</td>
                    <td style="border-right: solid 1px black;">4.54/ 5.31/ 0.26</td>
                    <td style="border-right: solid 1px black;">0.76/ 2.66/ 0.06</td>
                    <td>3.73/ 4.34/ 0.15</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">TI</td>
                    <td style="border-right: solid 1px black;">7.54/ 5.73/ 0.12</td>
                    <td style="border-right: solid 1px black;">4.86/ 4.81/ 0.14</td>
                    <td style="border-right: solid 1px black;">7.62/ 5.27/ 0.22</td>
                    <td style="border-right: solid 1px black;">1.80/ 3.60/ 0.08</td>
                    <td style="border-right: solid 1px black;">6.64/ 4.54/ 0.22</td>
                    <td style="border-right: solid 1px black;">3.92/ 4.43/ 0.42</td>
                    <td style="border-right: solid 1px black;">3.72/ 3.99/ 0.34</td>
                    <td>5.16/ 4.62/ 0.22</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">DI</td>
                    <td style="border-right: solid 1px black;">9.88/ 6.59/ 0.34</td>
                    <td style="border-right: solid 1px black;">14.24/ 7.01/ 0.68</td>
                    <td style="border-right: solid 1px black;"><b>20.10</b>/ <b>7.82</b>/ 0.82</td>
                    <td style="border-right: solid 1px black;">2.20/ 3.73/ 0.04</td>
                    <td style="border-right: solid 1px black;"><b>36.62</b>/ <b>9.40</b>/ <b>5.56</b></td>
                    <td style="border-right: solid 1px black;"><b>27.74</b>/ <b>9.49</b>/ <b>5.28</b></td>
                    <td style="border-right: solid 1px black;"><b>10.40</b>/ <b>5.45</b>/ <b>1.08</b></td>
                    <td>17.31/ 7.07/ <b>1.97</b></td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">SGM</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;"><b>21.62</b>/ 7.74/ <b>1.22</b></td>
                    <td style="border-right: solid 1px black;">16.22/ 6.95/ <b>0.98</b></td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td><b>18.92</b>/ <b>7.345</b>/ 1.71</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">LinBP</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">13.44/ <b>8.29</b>/ 0.52</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td>13.44/ 8.29/ 0.52</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">ILA</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">7.64/ 6.66/ 0.38</td>
                    <td style="border-right: solid 1px black;">1.98/ <b>5.52</b>/ 0.04</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td>4.81/ 6.09/ 0.21</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">IR</td>
                    <td style="border-right: solid 1px black;"><b>11.66</b>/ 6.62/ <b>0.52</b></td>
                    <td style="border-right: solid 1px black;">11.16/ 6.27/ 0.54</td>
                    <td style="border-right: solid 1px black;">15.64/ 7.35/ 0.84</td>
                    <td style="border-right: solid 1px black;">2.52/ 3.59/ 0.12</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td style="border-right: solid 1px black;">/</td>
                    <td>10.25/ 5.96/ 0.51</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">MI</td>
                    <td style="border-right: solid 1px black;">10.36/ 6.91/ 0.34</td>
                    <td style="border-right: solid 1px black;">8.60/ 6.51/ 0.38</td>
                    <td style="border-right: solid 1px black;">14.50/ 7.50/ 0.78</td>
                    <td style="border-right: solid 1px black;">2.28/ 4.04/ 0.10</td>
                    <td style="border-right: solid 1px black;">17.80/ 7.65/ 1.52</td>
                    <td style="border-right: solid 1px black;">14.88/ 8.12/ 1.34</td>
                    <td style="border-right: solid 1px black;">3.96/ 4.55/ 0.30</td>
                    <td>10.34/ 6.47/ 0.68</td>
                </tr>
                <tr>
                    <td style="border-right: solid 1px black;">NI</td>
                    <td style="border-right: solid 1px black;">10.62/ <b>6.92</b>/ 0.48</td>
                    <td style="border-right: solid 1px black;">8.66/ 6.53/ 0.46</td>
                    <td style="border-right: solid 1px black;">14.12/ 7.49/ 0.74</td>
                    <td style="border-right: solid 1px black;">2.38/ 4.01/ 0.10</td>
                    <td style="border-right: solid 1px black;">18.14/ 7.71/ 1.80</td>
                    <td style="border-right: solid 1px black;">14.54/ 8.11/ 1.32</td>
                    <td style="border-right: solid 1px black;">4.12/ 4.57/ 0.28</td>
                    <td>10.37/ 6.48/ 0.74</td>
                </tr>
                <tr style="border-bottom: solid 1px black;">
                    <td style="border-right: solid 1px black;">VMI</td>
                    <td style="border-right: solid 1px black;">10.34/ 6.84/ 0.28</td>
                    <td style="border-right: solid 1px black;">7.58/ 6.10/ 0.42</td>
                    <td style="border-right: solid 1px black;">13.20/ 7.15/ 0.60</td>
                    <td style="border-right: solid 1px black;"><b>3.38</b>/ 0.06/ <b>1.40</b></td>
                    <td style="border-right: solid 1px black;">9.40/ 5.87/ 0.60</td>
                    <td style="border-right: solid 1px black;">7.54/ 5.92/ 0.54</td>
                    <td style="border-right: solid 1px black;">3.86/ 4.47/ 0.32</td>
                    <td>7.9/ 5.20/ 0.59</td>
                </tr>
            </table>

<font size=1>
  
| Model | VGG16 | Res50 | Dense121 | IncV3 | ViT-B/16 | Deit-B/16 | Swin-B/4/7 | Avg |
| :----: | :----: | :----: | :----: | :----: | :---: | :---: | :------: | :------: |
| PGD | 4.74/ 5.49/ 0.06 | 2.68/ 4.22/ 0.10 | 5.52/ 5.19/ 0.16 | 0.62/ 2.42/ 0.04 | 7.22/ 5.11/ 0.40 | 4.54/ 5.31/ 0.26 | 0.76/ 2.66/ 0.06 | 3.73/ 4.34/ 0.15 |
| TI | 7.54/ 5.73/ 0.12 | 4.86/ 4.81/ 0.14 | 7.62/ 5.27/ 0.22 | 1.80/ 3.60/ 0.08 | 6.64/ 4.54/ 0.22 | 3.92/ 4.43/ 0.42 | 3.72/ 3.99/ 0.34 | 5.16/ 4.62/ 0.22 |
| DI | 9.88/ 6.59/ 0.34 | 14.24/ 7.01/ 0.68 | 20.10/ 7.82/ 0.82 | 2.20/ 3.73/ 0.04 | 36.62/ 9.40/ 5.56 | 27.74/ 9.49/ 5.28 | 10.40/ 5.45/ 1.08 | 17.31/ 7.07/ 1.97 |
| SGM | / | 21.62/ 7.74/ 1.22 | 16.22/ 6.95/ 0.98 | / | / | / | / | 18.92/ 7.345/ 1.71 |
| LinBP | / | 13.44/ 8.29/ 0.52 | / | / | / | / | / | 13.44/ 8.29/ 0.52 |
| ILA | / | / | 7.64/ 6.66/ 0.38 | 1.98/ 5.52/ 0.04 | / | / | / | 4.81/ 6.09/ 0.21 |
| IR | 11.66/ 6.62/ 0.52 | 11.16/ 6.27/ 0.54 | 15.64/ 7.35/ 0.84 | 2.52/ 3.59/ 0.12 | / | / | / | 10.25/ 5.96/ 0.51 |
| MI | 10.36/ 6.91/ 0.34 | 8.60/ 6.51/ 0.38 | 14.50/ 7.50/ 0.78 | 2.28/ 4.04/ 0.10 | 17.80/ 7.65/ 1.52 | 14.88/ 8.12/ 1.34 | 3.96/ 4.55/ 0.30 | 10.34/ 6.47/ 0.68 |
| NI | 10.62/ 6.92/ 0.48 | 8.66/ 6.53/ 0.46 | 14.12/ 7.49/ 0.74 | 2.38/ 4.01/ 0.10 | 18.14/ 7.71/ 1.80 | 14.54/ 8.11/ 1.32 | 4.12/ 4.57/ 0.28 | 10.37/ 6.48/ 0.74 |
| VMI | 10.34/ 6.84/ 0.28 | 7.58/ 6.10/ 0.42 | 13.20/ 7.15/ 0.60 | 3.38/ 0.06/ 1.40 | 9.40/ 5.87/ 0.60 | 7.54/ 5.92/ 0.54 | 3.86/ 4.47/ 0.32 | 7.9/ 5.20/ 0.59 |

</font>

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
