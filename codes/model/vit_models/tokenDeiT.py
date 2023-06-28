# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


import random
import timm
from torchvision import transforms

from .utils import *
from .tokenViT import TokenViT


def drop(x, pos, drop_pos_list, dim, drop_prob):
    '''
    Args:
        pos: 0-5 current position
        drop_pos_list: corresponds to whether this position is dropped
        dim: drop dimension, list, len = 6 => separated drop
    Returns:
    '''
    if dim is None or not drop_pos_list[pos] or drop_prob == 0.:
        return x

    prob = drop_prob

    selected_dim = None

    # if not len(dim) == 6 and 0 in dim:
    #     prob = 0.15
    #     prob = prob / len(dim)
    # elif not len(dim) == 6 and 1 in dim:
    #     prob = 0.01
    if  len(dim) != 6:
        prob = prob / len(dim)

    elif len(dim) == 6:
        prob = prob / sum([x != -1 for x in dim])
        selected_dim = dim[pos]

    keep_prob = 1 - prob
    if (len(dim) != 6 and 0 in dim) or (len(dim) == 6 and selected_dim == 0):
        shape = x.shape  # drop neuron
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    if (len(dim) != 6 and 1 in dim) or (len(dim) == 6 and selected_dim == 1):
        shape = (x.shape[0],) + (x.shape[1],) + (1,) * (x.ndim - 2)  # #drop token
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    if (len(dim) != 6 and 2 in dim ) or (len(dim) == 6 and selected_dim == 2):
        shape = (x.shape[0], 1,) + (x.shape[2],) + (1,) * (x.ndim -3)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        x = x.div(keep_prob) * random_tensor
    return x

def load_deit(device, model_name):
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    return model_official.to(device)



class TOKENDeiT(TokenViT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., num_perm = 0,
                 token_combi = 0, drop_dim=None, drop_prob=None, drop_pos=None, min = 8, max = 20, drop_first = 0):

        super().__init__( img_size=img_size, patch_size=patch_size, in_chans=in_chans, n_classes=n_classes,
                 embed_dim=embed_dim, depth=depth, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                          drop_dim = drop_dim, drop_prob= drop_prob, drop_pos= drop_pos)
        self.embed_dim = embed_dim
        self.num_classes = n_classes

        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        self.drop_first = drop_first

        self.drop_dim = drop_dim
        self.drop_pos = drop_pos
        self.drop_prob = drop_prob

        widths = list(range(min, max+1))
        self.transforms = [transforms.Resize((x * 16, x * 16)) for x in widths]
        self.token_combi = token_combi
        self.num_perm = num_perm
        print(min, max)
        self.training = False


    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        if '0' in self.token_combi:
            x = random.choice(self.transforms)(x.clone())

        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # dist_token = self.dist_token.expand(B, -1, -1) ##
        x = torch.cat((cls_tokens, x), dim=1)

        pos_emb = self.pos_embed

        if '0' in self.token_combi:
            pos_emb = interpolate_pos_encoding(self, x, w, h)  # (n_samples, 1 + n_patches, embed_dim)

        x = x + pos_emb
        x = self.pos_drop(x)
        x = drop(x, 0, [1, 0,0,0,0,0], dim = [0],  drop_prob=self.drop_first)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)
        return x


# load dataset => resize at runtime?
def load_TokenDeiT(model_name, args):
    custom_config = load_deit_config(model_name)
    model = TOKENDeiT(**custom_config, drop_prob = args.drop_prob, drop_dim=args.drop_dim, drop_pos = args.drop_pos,num_perm = args.num_perm,
                     token_combi = args.token_combi, min = args.min, max = args.max, drop_first = args.drop_first)
    model.eval()

    src_model_name = 'deit'+ args.src_model.split('deit')[-1]
    model_official = load_deit_official(src_model_name)
    print(type(model_official))

    assign_value(model_official, model)
    assert get_n_params(model) == get_n_params(model_official)

    return model.to(args.device), model_official.to(args.device)