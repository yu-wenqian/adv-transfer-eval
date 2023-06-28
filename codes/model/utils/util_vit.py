import torch
from torchvision import transforms, models


import os
import json
import xml.dom.minidom
import xml.etree.ElementTree as ET
from timm.models import create_model
from codes.model import vit_models
import timm
import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))



def get_mean_std(model_name):
    if 'vit' in model_name:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        return mean, std
    if 'deit' in model_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return mean, std

    if model_name in model_names:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return mean, std
    if 'T2t' in model_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return mean, std
    if 'tnt' in model_name:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        return mean, std

    if 'swin' in model_name or 'levit' in model_name or 'cait' in model_name:
        # https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return mean, std


def get_model(model_name, args):
    print(f"load {model_name}")
    mean, std = get_mean_std(model_name)

    # get the source model
    if model_name in model_names:
        model = models.__dict__[model_name](pretrained=True)
        model.eval()
        model_official = model

    elif 'swin' in model_name or 'levit' in model_name or 'cait' in model_name:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model_official = model
    # https://gitcode.net/mirrors/rwightman/pytorch-image-models/-/blob/master/timm/models/levit.py?from_codechina=yes
    elif 'ghostvit' in model_name:
        model, model_official = vit_models.ghostViT.load_ghostViT(model_name, args)

    elif 'idvit' in model_name:
        model, model_official = vit_models.idViT.load_idViT(args.device, model_name)

    elif 'tivit' in model_name:
        model, model_official = vit_models.tiViT.load_tiViT(args.device, model_name)

    elif 'tokenvit' in model_name:
        model, model_official = vit_models.tokenViT.load_TokenViT(args)

    elif 'sivit' in model_name:
        model, model_official = vit_models.siViT.load_siViT(args)

    elif 'resizevit' in model_name or 'resizedropTvit' in model_name \
            or 'dropTvit' in model_name or 'dropNvit' in model_name or \
            'dropCvit' in model_name or 'dropPvit' in model_name:
        model, model_official = vit_models.tokenViT.load_TokenViT(args)

    elif 'drop' in model_name and 'vit' in model_name:
        model, model_official = vit_models.dropViT.load_DropViT(args)

    elif model_name == 'PatchViT':
        model, model_official = vit_models.patchViT.load_patchViT(args)

    elif 'attnvit' in model_name:
        model, model_official = vit_models.attnViT.load_attnViT(args)

    elif 'msvit' in model_name:
        model, model_official = vit_models.msViT.load_msViT(args)

    elif 'vit' in model_name:
        model, model_official = vit_models.vit.load_ViT(model_name, args)


    elif 'ghostdeit' in model_name:
        model, model_official = vit_models.ghostDeiT.load_ghostDeiT(model_name, args)

    elif 'iddeit' in model_name:
        model, model_official = vit_models.idDeiT.load_idDeiT(model_name, args)

    elif 'tideit' in model_name:
        model, model_official = vit_models.tiDeiT.load_tiDeiT(model_name, args)

    elif 'resizedeit' in model_name or 'resizedropTdeit' in model_name \
            or 'dropTdeit' in model_name or 'dropNdeit' in model_name\
            or 'dropPdeit' in model_name or 'dropCdeit' in model_name:
        model, model_official = vit_models.tokenDeiT.load_TokenDeiT(model_name, args)

    elif 'deit' in model_name:
        model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        model.training = False
        model.eval()
        model_official = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
        model_official.training = False
        model_official.eval()

    elif 'T2t' in model_name or 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        model_official = model

    elif 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        model_official = model

    else:
        raise NotImplementedError(f"Please provide correct model names: {model_names}")

    return (model.to(args.device), model_official.to(args.device)), mean, std